# SPDX-License-Identifier: Apache-2.0
# Modified multi-round QA benchmark with:
#   - Normal distribution based time gap sampling (--gap-mean, --gap-std)
#   - Poisson distribution based output length sampling (--output-lambda, --output-min, --output-max)
#   - RequestMonitor for event logging (monitor.jsonl)
#   - MetricsScraper for LMCache Prometheus metrics (metrics.jsonl)
#   - Gap history / sampled output length tracking in summary.csv

from dataclasses import dataclass, field
from logging import Logger
from typing import Optional
import argparse
import asyncio
import json
import logging
import os
import re
import threading
import time
import urllib.request

import numpy as np
import openai
import pandas as pd


# ========================== Logger (from utils.py) ==========================

def build_format(color):
    reset = "\x1b[0m"
    underline = "\x1b[3m"
    return (
        f"{color}[%(asctime)s] %(levelname)s:{reset} %(message)s "
        + f"{underline}(%(filename)s:%(lineno)d:%(name)s){reset}"
    )


class CustomFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: build_format("\x1b[1m"),
        logging.INFO: build_format("\x1b[32;20m"),
        logging.WARNING: build_format("\x1b[33;20m"),
        logging.ERROR: build_format("\x1b[31;20m"),
        logging.CRITICAL: build_format("\x1b[31;1m"),
    }

    def format(self, record):
        fmt = self.FORMATS.get(record.levelno)
        return logging.Formatter(fmt).format(record)


def init_logger(name: str, log_level=logging.DEBUG) -> Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    return logger


logger = init_logger(__name__, logging.INFO)


# ====================== AsyncLoopWrapper (from utils.py) ======================

class AsyncLoopWrapper:
    _loop: asyncio.AbstractEventLoop | None = None
    _thread: threading.Thread | None = None
    _logger = init_logger("AsyncLoopWrapper")

    @classmethod
    def WaitLoop(cls):
        assert cls._loop is not None
        async def wait_for_tasks():
            current = asyncio.current_task(cls._loop)
            tasks = [t for t in asyncio.all_tasks(cls._loop) if not t.done() and t is not current]
            cls._logger.info(f"Waiting for {len(tasks)} tasks to finish")
            if tasks:
                await asyncio.gather(*tasks)
        future = asyncio.run_coroutine_threadsafe(wait_for_tasks(), cls._loop)
        try:
            future.result()
        except Exception as e:
            cls._logger.error(f"Error while waiting for tasks: {e}")

    @classmethod
    def StartLoop(cls):
        if cls._loop is not None:
            return
        cls._loop = asyncio.new_event_loop()
        def run_loop():
            asyncio.set_event_loop(cls._loop)
            cls._loop.run_forever()
        cls._thread = threading.Thread(target=run_loop)
        cls._thread.start()

    @classmethod
    def StopLoop(cls):
        assert cls._loop is not None and cls._thread is not None
        cls.WaitLoop()
        cls._loop.call_soon_threadsafe(cls._loop.stop)
        cls._thread.join()

    @classmethod
    def GetOrStartLoop(cls) -> asyncio.AbstractEventLoop:
        if cls._loop is None:
            cls.StartLoop()
        return cls._loop


# ========================== RequestMonitor ==========================

class RequestMonitor:
    """Request launch/finish 이벤트를 JSONL 파일로 기록"""

    def __init__(self, log_path: str = "monitor.jsonl"):
        self.log_path = log_path
        self.events: list[dict] = []
        # 시작 시 파일 초기화
        with open(self.log_path, "w") as f:
            pass

    def on_request_launch(self, user_id: int, question_id: int, timestamp: float):
        self.events.append({
            "event": "launch",
            "user_id": user_id,
            "question_id": question_id,
            "timestamp": timestamp,
        })

    def on_request_finish(self, user_id: int, question_id: int, timestamp: float,
                          ttft: float, prompt_tokens: int, gen_tokens: int):
        self.events.append({
            "event": "finish",
            "user_id": user_id,
            "question_id": question_id,
            "timestamp": timestamp,
            "ttft": ttft,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
        })

    def on_gap_sample(self, user_id: int, gap: float, timestamp: float):
        self.events.append({
            "event": "gap_sample",
            "user_id": user_id,
            "gap": gap,
            "timestamp": timestamp,
        })

    def flush(self):
        if not self.events:
            return
        with open(self.log_path, "a") as f:
            for ev in self.events:
                f.write(json.dumps(ev) + "\n")
        self.events.clear()


# ========================== MetricsScraper ==========================

# Prometheus metric 이름 → 수집할 것들
# LMCache metrics (lmcache: prefix)
LMCACHE_METRICS = [
    "lmcache:retrieve_hit_rate",
    "lmcache:lookup_hit_rate",
    "lmcache:num_hit_tokens",
    "lmcache:num_requested_tokens",
    "lmcache:num_retrieve_requests",
    "lmcache:num_store_requests",
    "lmcache:local_cache_usage",
    "lmcache:local_cpu_evict_count",
]
# vLLM built-in cache metrics (vllm 0.16+)
VLLM_CACHE_METRICS = [
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:num_preemptions_total",
    "vllm:kv_cache_usage_perc",
]
ALL_SCRAPE_METRICS = LMCACHE_METRICS + VLLM_CACHE_METRICS

# Diverse question templates for realistic workload
DIVERSE_QUESTIONS = [
    "Based on the context above, explain the main concepts in detail.",
    "What are the key takeaways from the provided text?",
    "Can you analyze and summarize the most important aspects?",
    "Provide a comprehensive overview of the topics discussed.",
    "What patterns or themes do you notice in the context?",
    "Describe the most significant points from the text.",
    "How would you explain the main ideas to someone unfamiliar with the topic?",
    "What conclusions can you draw from the information provided?",
    "Identify and elaborate on the critical details in the context.",
    "What is your analysis of the key arguments presented above?",
]


class MetricsScraper:
    """vLLM /metrics 엔드포인트에서 LMCache Prometheus metric을 주기적으로 수집"""

    def __init__(self, metrics_url: str, log_path: str = "metrics.jsonl",
                 interval: float = 5.0):
        self.metrics_url = metrics_url
        self.log_path = log_path
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # 시작 시 파일 초기화
        with open(self.log_path, "w") as f:
            pass

    def _parse_prometheus(self, text: str) -> dict:
        """Prometheus text format에서 LMCache + vLLM cache metric 파싱"""
        result = {}
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # "metric_name{labels} value" 또는 "metric_name value" 형태
            for metric_name in ALL_SCRAPE_METRICS:
                if line.startswith(metric_name):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            result[metric_name] = float(parts[-1])
                        except ValueError:
                            pass
            # vllm:prompt_tokens_by_source_total{source="xxx"} 별도 파싱
            if line.startswith("vllm:prompt_tokens_by_source_total"):
                m = re.search(r'source="(\w+)"', line)
                parts = line.split()
                if m and len(parts) >= 2:
                    try:
                        result[f"vllm:prompt_tokens_source_{m.group(1)}"] = float(parts[-1])
                    except ValueError:
                        pass
        return result

    def _scrape_once(self) -> dict | None:
        try:
            req = urllib.request.Request(self.metrics_url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                text = resp.read().decode("utf-8")
            return self._parse_prometheus(text)
        except Exception:
            return None

    def _loop(self):
        while not self._stop_event.is_set():
            metrics = self._scrape_once()
            if metrics:
                record = {"timestamp": time.time(), **metrics}
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            self._stop_event.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_final_snapshot(self) -> dict | None:
        """벤치마크 종료 시 마지막 metric 한번 더 수집"""
        return self._scrape_once()


# ========================== Data Classes ==========================

@dataclass
class WorkloadConfig:
    num_users: int
    system_prompt_len: int
    user_info_len: int
    answer_len: int          # max cap (used when output_lambda is None)
    num_rounds: int
    qps: float
    model: str
    enable_user_id: bool

    # --- NEW: Normal distribution gap ---
    gap_mean: Optional[float] = None   # None → QPS 기반 자동 계산
    gap_std: float = 0.0

    # --- NEW: Poisson output length ---
    output_lambda: Optional[float] = None  # None → answer_len 고정 사용
    output_min: int = 1
    output_max: int = 2048

    # --- NEW: Dry-run simulated delay ---
    dry_run_delay: float = 1.0  # simulated request duration in dry-run (seconds)

    enforce_strict_concurrent_users: bool = False
    disable_ramp_up: bool = False

    # --- NEW: Realistic workload ---
    shared_prefix_ratio: Optional[float] = None  # None → legacy "hi" mode
    context_file: Optional[str] = None


@dataclass
class UserConfig:
    user_id: int
    system_prompt_len: int
    user_info_len: int
    answer_len: int
    num_rounds: int
    enable_user_id: bool

    # --- NEW: gap distribution params ---
    gap_mean: float = 1.0
    gap_std: float = 0.0

    # --- NEW: output length distribution params ---
    output_lambda: Optional[float] = None
    output_min: int = 1
    output_max: int = 2048

    # --- NEW: dry-run simulated delay ---
    dry_run_delay: float = 1.0

    @staticmethod
    def new_user_config(user_id: int, wc: WorkloadConfig) -> "UserConfig":
        if wc.gap_mean is not None:
            gap_mean = wc.gap_mean
            gap_std = wc.gap_std
        else:
            gap_mean = wc.num_users / wc.qps
            gap_std = 0.0

        return UserConfig(
            user_id=user_id,
            system_prompt_len=wc.system_prompt_len,
            user_info_len=wc.user_info_len,
            answer_len=wc.answer_len,
            num_rounds=wc.num_rounds,
            enable_user_id=wc.enable_user_id,
            gap_mean=gap_mean,
            gap_std=gap_std,
            output_lambda=wc.output_lambda,
            output_min=wc.output_min,
            output_max=wc.output_max,
            dry_run_delay=wc.dry_run_delay,
        )


# ========================== ChatHistory ==========================

class ChatHistory:
    def __init__(self):
        self.history = []

    def on_user_query(self, query: str):
        if len(self.history) == 0:
            self.history.append({"role": "user", "content": query})
        else:
            assert self.history[-1]["role"] == "assistant"
            self.history.append({"role": "user", "content": query})

    def on_system_response(self, response: str):
        assert len(self.history) > 0 and self.history[-1]["role"] == "user"
        self.history.append({"role": "assistant", "content": response})

    def get_messages_for_openai(self):
        return self.history

    def __len__(self):
        return len(self.history)


# ========================== Response ==========================

@dataclass
class Response:
    body: str
    ttft: float
    generation_time: float
    prompt_tokens: int
    generation_tokens: int
    launch_time: float
    finish_time: float


# ========================== RequestExecutor ==========================

class RequestExecutor:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.loop = AsyncLoopWrapper.GetOrStartLoop()

    async def _async_launch_request(self, messages, max_tokens, extra_headers=None):
        start_time = time.time()
        first_token_time = None
        words = ""

        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0,
            stream=True,
            max_tokens=max_tokens,
            stream_options={"include_usage": True},
            extra_headers=extra_headers,
        )

        async for tok in response:
            if not tok.choices:
                continue
            chunk = tok.choices[0].delta.content
            if chunk is not None:
                if first_token_time is None and chunk != "":
                    first_token_time = time.time()
                words += chunk

        tokens_out = tok.usage.completion_tokens
        tokens_prefill = tok.usage.prompt_tokens

        return Response(
            body=words,
            ttft=first_token_time - start_time if first_token_time else 0.0,
            generation_time=time.time() - first_token_time if first_token_time else 0.0,
            prompt_tokens=tokens_prefill,
            generation_tokens=tokens_out,
            launch_time=start_time,
            finish_time=time.time(),
        )

    def launch_request(self, chat_history, max_tokens, finish_callback, extra_headers=None):
        messages = chat_history.get_messages_for_openai()
        future = asyncio.run_coroutine_threadsafe(
            self._async_launch_request(messages, max_tokens, extra_headers),
            self.loop,
        )
        future.add_done_callback(lambda x: finish_callback(x.result()))


# ========================== UserSession ==========================

class UserSession:
    def __init__(self, user_config: UserConfig, monitor: Optional[RequestMonitor] = None,
                 use_sharegpt=False, sharegpt_data=None,
                 context_words=None, shared_prefix_ratio=None):
        self.user_config = user_config
        self.monitor = monitor
        self.last_request_time: float | None = None
        self.chat_history = ChatHistory()
        self.question_id = 0
        self.use_sharegpt = use_sharegpt
        if self.use_sharegpt:
            self.sharegpt_data = sharegpt_data
            self.start_with_gpt = sharegpt_data["num_round"] % 2 == 0

        # Realistic workload text
        self.context_words = context_words or []
        self.shared_prefix_ratio = shared_prefix_ratio
        self._use_realistic_text = (shared_prefix_ratio is not None and len(self.context_words) > 0)

        self.has_unfinished_request = False
        self.last_unfinished_log = 0.0

        # Result tracking
        self.prompt_lengths: list[int] = []
        self.generation_lengths: list[int] = []
        self.ttfts: list[float] = []
        self.generation_times: list[float] = []
        self.launch_times: list[float] = []
        self.finish_times: list[float] = []

        # NEW: gap & output length tracking
        self.gap_history: list[float] = []
        self.sampled_output_lengths: list[int] = []
        self.current_gap: float = self._sample_gap()

        self.finished = False
        self._dry_run_timer: threading.Timer | None = None

    def _sample_gap(self) -> float:
        """정규분포 N(mean, std)에서 gap sampling. std=0이면 고정값."""
        if self.user_config.gap_std == 0:
            return self.user_config.gap_mean
        gap = np.random.normal(self.user_config.gap_mean, self.user_config.gap_std)
        return max(gap, 0.1)

    def _sample_output_length(self) -> int:
        """Poisson(lambda)에서 output length sampling. lambda=None이면 answer_len 고정."""
        if self.user_config.output_lambda is None:
            return self.user_config.answer_len
        length = np.random.poisson(self.user_config.output_lambda)
        return int(np.clip(length, self.user_config.output_min, self.user_config.output_max))

    def _update_result(self, response: Response):
        self.prompt_lengths.append(response.prompt_tokens)
        self.generation_lengths.append(response.generation_tokens)
        self.ttfts.append(response.ttft)
        self.generation_times.append(response.generation_time)
        self.launch_times.append(response.launch_time)
        self.finish_times.append(response.finish_time)

    def _build_system_prompt(self):
        if self._use_realistic_text:
            return self._build_realistic_prompt()
        # Legacy: dummy "hi" tokens (backward compatible)
        dummy_sys = " ".join(["hi"] * self.user_config.system_prompt_len)
        dummy_user = " ".join(["hi"] * self.user_config.user_info_len)
        return (
            f"Hi, here's some system prompt: {dummy_sys}."
            f"For user {self.user_config.user_id}, "
            f"here are some other context: {dummy_user}."
        )

    def _build_realistic_prompt(self):
        """Build prompt with shared prefix + per-user unique text from real content."""
        total_words = self.user_config.system_prompt_len + self.user_config.user_info_len
        shared_count = int(total_words * self.shared_prefix_ratio)
        unique_count = total_words - shared_count

        # Shared portion: same for all users → LMCache hit
        shared_part = " ".join(self.context_words[:shared_count])

        # Unique portion: different offset per user → LMCache miss
        if unique_count > 0:
            remaining = self.context_words[shared_count:]
            if not remaining:
                remaining = self.context_words
            # Prime stride ensures different starting points per user
            offset = (self.user_config.user_id * 997) % len(remaining)
            unique_words = [remaining[(offset + i) % len(remaining)] for i in range(unique_count)]
            unique_part = " ".join(unique_words)
        else:
            unique_part = ""

        parts = []
        if shared_part:
            parts.append(f"You are a helpful assistant. Here is the shared context:\n\n{shared_part}")
        if unique_part:
            parts.append(f"\n\nHere is additional context for user {self.user_config.user_id}:\n\n{unique_part}")

        return "\n".join(parts) + "\n\n"

    def _build_new_question(self):
        self.question_id += 1
        if self._use_realistic_text:
            idx = (self.user_config.user_id * 7 + self.question_id) % len(DIVERSE_QUESTIONS)
            return f"Question #{self.question_id}: {DIVERSE_QUESTIONS[idx]}"
        return f"Here's question #{self.question_id}: can you tell me a new long story with a happy ending?"

    def _launch_new_request(self, timestamp: float, request_executor: Optional[RequestExecutor]):
        # Build prompt
        if self.use_sharegpt:
            idx = 2 * self.question_id + (1 if self.start_with_gpt else 0)
            prompt = self.sharegpt_data["conversations"][idx]["value"]
            self.question_id += 1
        else:
            prompt = self._build_new_question()

        if len(self.chat_history) == 0:
            prompt = self._build_system_prompt() + prompt

        self.chat_history.on_user_query(prompt)
        logger.debug(f"User {self.user_config.user_id} issues request {self.question_id}")

        # Sample output length
        sampled_len = self._sample_output_length()
        self.sampled_output_lengths.append(sampled_len)

        if self.use_sharegpt:
            idx = 2 * self.question_id if self.start_with_gpt else 2 * self.question_id - 1
            max_tokens = min(self.sharegpt_data["conversations"][idx]["num_tokens"], sampled_len)
        else:
            max_tokens = sampled_len

        # Monitor: launch event
        if self.monitor:
            self.monitor.on_request_launch(self.user_config.user_id, self.question_id, timestamp)

        if request_executor is not None:
            request_executor.launch_request(
                self.chat_history,
                max_tokens,
                self._on_request_finished,
                extra_headers={"x-user-id": str(self.user_config.user_id)},
            )
            self.has_unfinished_request = True
        else:
            # Dry-run: simulate delayed completion via timer
            self.has_unfinished_request = True
            delay = self.user_config.dry_run_delay
            launch_t = timestamp
            uid = self.user_config.user_id
            qid = self.question_id
            mt = max_tokens

            def _delayed_finish():
                finish_t = time.time()
                self.chat_history.on_system_response("")
                sim_ttft = delay * 0.1  # simulate 10% of delay as TTFT
                simulated = Response(
                    body="",
                    ttft=sim_ttft,
                    generation_time=delay - sim_ttft,
                    prompt_tokens=0,
                    generation_tokens=mt,
                    launch_time=launch_t,
                    finish_time=finish_t,
                )
                self._update_result(simulated)
                self.has_unfinished_request = False
                if self.monitor:
                    self.monitor.on_request_finish(uid, qid, finish_t, sim_ttft, 0, mt)

            self._dry_run_timer = threading.Timer(delay, _delayed_finish)
            self._dry_run_timer.start()

        self.last_request_time = timestamp

    def _on_request_finished(self, response: Response):
        self.chat_history.on_system_response(response.body)
        self.has_unfinished_request = False
        logger.debug(
            f"User {self.user_config.user_id} finished. "
            f"Prompt: {response.prompt_tokens}, Gen: {response.generation_tokens}"
        )
        self._update_result(response)
        if self.monitor:
            self.monitor.on_request_finish(
                self.user_config.user_id, self.question_id, response.finish_time,
                response.ttft, response.prompt_tokens, response.generation_tokens,
            )

    def set_internal_state(self, offset: float, timestamp: float):
        assert len(self.chat_history) == 0
        num_passed = int(offset / self.user_config.gap_mean) + 1
        passed_time = (num_passed - 1) * self.user_config.gap_mean
        self.last_request_time = timestamp - offset + passed_time
        self.question_id = num_passed

    def step(self, timestamp: float, request_executor: Optional[RequestExecutor]):
        if self.question_id >= self.user_config.num_rounds and not self.has_unfinished_request:
            self.finished = True
            return

        if self.last_request_time is None:
            self._launch_new_request(timestamp, request_executor)
            # Sample next gap
            self.current_gap = self._sample_gap()
            self.gap_history.append(self.current_gap)
            if self.monitor:
                self.monitor.on_gap_sample(self.user_config.user_id, self.current_gap, timestamp)
            return

        if timestamp - self.last_request_time > self.current_gap:
            if self.has_unfinished_request:
                if timestamp - self.last_unfinished_log > 10:
                    logger.warning(
                        f"User {self.user_config.user_id} has unfinished request, can't keep up."
                    )
                    self.last_unfinished_log = timestamp
                return

            self._launch_new_request(timestamp, request_executor)
            # Sample next gap
            self.current_gap = self._sample_gap()
            self.gap_history.append(self.current_gap)
            if self.monitor:
                self.monitor.on_gap_sample(self.user_config.user_id, self.current_gap, timestamp)
            return

    def summary(self) -> pd.DataFrame:
        n = len(self.prompt_lengths)
        # Pad gap_history and sampled_output_lengths to match length
        gaps = self.gap_history[:n] + [None] * max(0, n - len(self.gap_history))
        outs = self.sampled_output_lengths[:n] + [None] * max(0, n - len(self.sampled_output_lengths))

        df = pd.DataFrame({
            "prompt_tokens": self.prompt_lengths,
            "generation_tokens": self.generation_lengths,
            "ttft": self.ttfts,
            "generation_time": self.generation_times,
            "user_id": self.user_config.user_id,
            "question_id": list(range(1, n + 1)),
            "launch_time": self.launch_times,
            "finish_time": self.finish_times,
            "actual_gap": gaps,
            "sampled_output_len": outs,
        })
        return df


# ========================== UserSessionManager ==========================

class UserSessionManager:
    def __init__(self, workload_config: WorkloadConfig, monitor: Optional[RequestMonitor] = None,
                 init_user_id=0, use_sharegpt=False):
        self.workload_config = workload_config
        self.monitor = monitor
        self.sessions: list[UserSession] = []

        # Load realistic text if shared_prefix_ratio is set
        self.context_words = []
        self.shared_prefix_ratio = workload_config.shared_prefix_ratio
        if self.shared_prefix_ratio is not None:
            self._load_context_text(workload_config.context_file)

        # gap_mean 결정
        if workload_config.gap_mean is not None:
            effective_gap = workload_config.gap_mean
        else:
            effective_gap = workload_config.num_users / workload_config.qps

        session_alive_time = effective_gap * (workload_config.num_rounds - 1)
        self.gap_between_users = session_alive_time / (workload_config.num_users + 0)
        self.ramp_up_time = workload_config.num_users * self.gap_between_users

        logger.info(
            f"Gap between users: {self.gap_between_users:.2f}s\n"
            f"Effective gap per user (mean): {effective_gap:.2f}s\n"
            f"Gap std: {workload_config.gap_std:.2f}s\n"
            f"Expected session length: {session_alive_time:.2f}s"
        )
        if workload_config.output_lambda is not None:
            logger.info(
                f"Output length: Poisson(lambda={workload_config.output_lambda}), "
                f"clipped to [{workload_config.output_min}, {workload_config.output_max}]"
            )
        else:
            logger.info(f"Output length: fixed at {workload_config.answer_len}")

        self.user_id = init_user_id
        self.last_user_join = 0.0
        self.session_summaries: list[pd.DataFrame] = []
        self.start_time: float | None = None
        self.need_ramp_up = not workload_config.disable_ramp_up

        self.use_sharegpt = use_sharegpt
        if self.use_sharegpt:
            self._load_sharegpt_data()

        self.enforce_strict_concurrent_users = workload_config.enforce_strict_concurrent_users

    def _load_context_text(self, context_file):
        """Load text file for realistic prompt generation."""
        if context_file is None:
            # Default: man-bash.txt in same directory
            context_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "man-bash.txt")
        if not os.path.exists(context_file):
            logger.warning(f"Context file not found: {context_file}, falling back to legacy mode")
            self.shared_prefix_ratio = None
            return
        with open(context_file, "r") as f:
            text = f.read()
        self.context_words = text.split()
        total_needed = self.workload_config.system_prompt_len + self.workload_config.user_info_len
        logger.info(
            f"Realistic text mode: ratio={self.shared_prefix_ratio:.2f}, "
            f"words loaded={len(self.context_words)}, words needed/user={total_needed}, "
            f"shared={int(total_needed * self.shared_prefix_ratio)}, "
            f"unique={total_needed - int(total_needed * self.shared_prefix_ratio)}"
        )

    def _load_sharegpt_data(self):
        with open("ShareGPT.json", "r", encoding="utf-8") as f:
            self.sharegpt_data = json.load(f)
        self.sharegpt_data = [
            d for d in self.sharegpt_data
            if d["num_round"] > 2 * self.workload_config.num_rounds
        ]
        assert len(self.sharegpt_data) >= self.workload_config.num_users

    def _ramp_up(self, timestamp: float, ramp_up_time: float):
        for i in range(self.workload_config.num_users):
            session = self._create_user_session()
            offset = ramp_up_time - i * self.gap_between_users
            if offset < 0:
                break
            session.set_internal_state(offset, timestamp)
        self.need_ramp_up = False

    def _create_user_session(self):
        self.user_id += 1
        uc = UserConfig.new_user_config(self.user_id, self.workload_config)
        if self.use_sharegpt:
            session = UserSession(uc, self.monitor, True, self.sharegpt_data[self.user_id])
        else:
            session = UserSession(
                uc, self.monitor,
                context_words=self.context_words,
                shared_prefix_ratio=self.shared_prefix_ratio,
            )
        self.sessions.append(session)
        return session

    def _remove_finished_sessions(self):
        finished = [s for s in self.sessions if s.finished]
        if finished:
            logger.info(
                f"Removing {len(finished)} finished sessions, "
                f"active: {len(self.sessions) - len(finished)}"
            )
            for s in finished:
                self.session_summaries.append(s.summary())
        self.sessions = [s for s in self.sessions if not s.finished]

    def _can_join_user(self, timestamp: float) -> bool:
        if timestamp - self.last_user_join <= self.gap_between_users:
            return False
        if self.enforce_strict_concurrent_users and len(self.sessions) >= self.workload_config.num_users:
            return False
        return True

    def step(self, timestamp: float, executor: Optional[RequestExecutor]):
        if self.need_ramp_up:
            self._ramp_up(timestamp, self.ramp_up_time)

        if self.start_time is None:
            self.start_time = timestamp

        if self._can_join_user(timestamp):
            self._create_user_session()
            self.last_user_join = timestamp
            logger.info(f"Joined user {self.user_id}, active: {len(self.sessions)}")

        for session in self.sessions:
            session.step(timestamp, executor)

        self._remove_finished_sessions()

    @staticmethod
    def ProcessSummary(df, start_time=None, end_time=None, pending_queries=0, config_qps=None):
        if start_time and end_time:
            launched = len(df.query(f"{start_time} <= launch_time <= {end_time}"))
            df = df.query(f"{start_time} <= finish_time <= {end_time}")
        else:
            launched = len(df)

        if config_qps is None:
            config_qps = 0.0
        if start_time is None:
            start_time = df["launch_time"].min()
        if end_time is None:
            end_time = df["finish_time"].max()

        total_time = end_time - start_time
        if total_time <= 0:
            logger.warning("No time elapsed in summary window")
            return df

        actual_qps = (launched + pending_queries) / total_time
        finished_qps = len(df) / total_time
        avg_ttft = df["ttft"].mean()

        total_gen = df["generation_tokens"].sum()
        avg_gen_speed = total_gen / total_time
        avg_gen_per_req = (df["generation_tokens"] / df["generation_time"].replace(0, np.nan)).mean()

        print("\n==================== Performance summary ======================")
        print(f"  Config QPS: {config_qps:.4f} reqs/s")
        print(f"  Actual QPS: {actual_qps:.4f} reqs/s")
        print(f"  Processing speed: {finished_qps:.4f} reqs/s")
        print(f"  Requests on-the-fly: {pending_queries}")
        print(f"  Output tokens/s: {avg_gen_speed:.4f}")
        print(f"  Avg gen throughput/req: {avg_gen_per_req:.4f} tokens/req/s")
        print(f"  Avg TTFT: {avg_ttft:.4f}s")

        if "actual_gap" in df.columns:
            gaps = df["actual_gap"].dropna()
            if len(gaps) > 0:
                print(f"  Gap stats: mean={gaps.mean():.3f}s, std={gaps.std():.3f}s, "
                      f"min={gaps.min():.3f}s, max={gaps.max():.3f}s")

        if "sampled_output_len" in df.columns:
            outs = df["sampled_output_len"].dropna()
            if len(outs) > 0:
                print(f"  Output len stats: mean={outs.mean():.1f}, std={outs.std():.1f}, "
                      f"min={outs.min():.0f}, max={outs.max():.0f}")

        print(f"  Time range: {total_time:.2f}s")
        print("===============================================================\n")
        return df

    def summary(self, start_time: float, end_time: float) -> pd.DataFrame:
        if not self.session_summaries and not self.sessions:
            return pd.DataFrame()

        dfs = self.session_summaries + [s.summary() for s in self.sessions]
        df = pd.concat(dfs, ignore_index=True)
        pending = len([s for s in self.sessions if s.has_unfinished_request])
        assert self.start_time is not None
        start_time = max(self.start_time, start_time)

        if self.workload_config.gap_mean is not None:
            qps = self.workload_config.num_users / self.workload_config.gap_mean
        else:
            qps = self.workload_config.qps

        df = UserSessionManager.ProcessSummary(df, start_time, end_time, pending, qps)
        return df


# ========================== Engine Warmup ==========================

def warmup_engine(executor):
    logger.info("Warming up the engine")
    for i in range(10):
        ch = ChatHistory()
        ch.on_user_query(f"WARMUP: Hi, I'm user {i}. Here are some text: {'hi ' * 100}.")
        executor.launch_request(ch, 100, lambda x: None)
    AsyncLoopWrapper.WaitLoop()


# ========================== CLI ==========================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Multi-round QA benchmark (v2)")

    # Original args
    parser.add_argument("--num-users", type=int, required=True)
    parser.add_argument("--shared-system-prompt", type=int, required=True,
                        help="Length of shared system prompt (tokens)")
    parser.add_argument("--user-history-prompt", type=int, required=True,
                        help="Length of user-specific history prompt (tokens)")
    parser.add_argument("--answer-len", type=int, default=256,
                        help="Max answer length (used when --output-lambda is not set)")
    parser.add_argument("--num-rounds", type=int, required=True)
    parser.add_argument("--qps", type=float, default=1.0,
                        help="Overall QPS (ignored if --gap-mean is set)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--time", type=int, default=None,
                        help="Duration in seconds")
    parser.add_argument("--output", type=str, default="summary.csv")
    parser.add_argument("--init-user-id", type=int, default=0)
    parser.add_argument("--request-with-user-id", action="store_true")
    parser.add_argument("--log-interval", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sharegpt", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enforce-strict-concurrent-users", action="store_true")
    parser.add_argument("--disable-ramp-up", action="store_true")

    # NEW: Realistic workload
    parser.add_argument("--shared-prefix-ratio", type=float, default=None,
                        help="Fraction of prompt shared across users (0.0-1.0). "
                             "When set, uses real text instead of dummy 'hi' tokens. "
                             "0.0=all unique, 0.3=chatbot, 0.5=RAG, 1.0=all shared.")
    parser.add_argument("--context-file", type=str, default=None,
                        help="Text file for realistic prompts (default: man-bash.txt)")

    # NEW: Normal distribution gap
    parser.add_argument("--gap-mean", type=float, default=None,
                        help="Mean of N(mean,std) for gap between requests (seconds). "
                             "Overrides QPS-based gap.")
    parser.add_argument("--gap-std", type=float, default=0.0,
                        help="Std of N(mean,std) for gap. 0=fixed, >0=stochastic.")

    # NEW: Poisson output length
    parser.add_argument("--output-lambda", type=float, default=None,
                        help="Lambda for Poisson output length sampling. "
                             "If not set, uses --answer-len as fixed value.")
    parser.add_argument("--output-min", type=int, default=1,
                        help="Min output length (clip)")
    parser.add_argument("--output-max", type=int, default=2048,
                        help="Max output length (clip)")

    # NEW: Dry-run delay
    parser.add_argument("--dry-run-delay", type=float, default=1.0,
                        help="Simulated request duration in dry-run mode (seconds). "
                             "Default 1.0s. Makes dry-run behave like real requests.")

    # NEW: Monitor
    parser.add_argument("--monitor-log", type=str, default="monitor.jsonl",
                        help="Path to monitor event log (JSONL)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip auto-generating plots after benchmark")

    # NEW: LMCache Metrics Scraper
    parser.add_argument("--metrics-url", type=str, default="http://localhost:8000/metrics",
                        help="vLLM /metrics endpoint URL for LMCache Prometheus metrics")
    parser.add_argument("--metrics-log", type=str, default="metrics.jsonl",
                        help="Path to LMCache metrics log (JSONL)")
    parser.add_argument("--metrics-interval", type=float, default=5.0,
                        help="Seconds between metric scrapes")
    parser.add_argument("--no-metrics", action="store_true",
                        help="Disable LMCache metrics scraping")

    # Process existing summary
    parser.add_argument("--process-summary", type=str, default=None,
                        help="Process an existing summary CSV instead of running benchmark")

    return parser.parse_args()


# ========================== Main ==========================

def main():
    args = parse_arguments()

    if args.process_summary:
        UserSessionManager.ProcessSummary(pd.read_csv(args.process_summary), pending_queries=0)
        return

    if args.verbose:
        global logger
        logger = init_logger(__name__, logging.DEBUG)

    monitor = RequestMonitor(args.monitor_log)

    # LMCache Metrics Scraper
    scraper = None
    if not args.no_metrics and not args.dry_run:
        scraper = MetricsScraper(
            metrics_url=args.metrics_url,
            log_path=args.metrics_log,
            interval=args.metrics_interval,
        )

    executor = None
    if not args.dry_run:
        executor = RequestExecutor(base_url=args.base_url, api_key="EMPTY", model=args.model)
        warmup_engine(executor)

    wc = WorkloadConfig(
        num_users=args.num_users,
        system_prompt_len=args.shared_system_prompt,
        user_info_len=args.user_history_prompt,
        answer_len=args.answer_len,
        num_rounds=args.num_rounds,
        qps=args.qps,
        model=args.model,
        enable_user_id=args.request_with_user_id,
        gap_mean=args.gap_mean,
        gap_std=args.gap_std,
        output_lambda=args.output_lambda,
        output_min=args.output_min,
        output_max=args.output_max,
        dry_run_delay=args.dry_run_delay,
        enforce_strict_concurrent_users=args.enforce_strict_concurrent_users,
        disable_ramp_up=args.disable_ramp_up,
        shared_prefix_ratio=args.shared_prefix_ratio,
        context_file=args.context_file,
    )

    manager = UserSessionManager(
        wc, monitor=monitor, init_user_id=args.init_user_id, use_sharegpt=args.sharegpt,
    )

    if scraper:
        scraper.start()
        logger.info(f"Metrics scraper started: {args.metrics_url} every {args.metrics_interval}s")

    step_interval = 0.1
    start_time = time.time()
    last_summary_time = start_time
    last_flush_time = start_time

    try:
        while True:
            manager.step(time.time(), executor)
            time.sleep(step_interval)

            now = time.time()
            # Periodic summary
            if now - last_summary_time > args.log_interval:
                manager.summary(last_summary_time, now)
                last_summary_time = now

            # Periodic monitor flush (every 5s)
            if now - last_flush_time > 5.0:
                monitor.flush()
                last_flush_time = now

            if args.time is not None and now - start_time > args.time:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted, collecting final results")

    if executor is not None:
        AsyncLoopWrapper.StopLoop()

    # Stop metrics scraper and collect final snapshot
    if scraper:
        final_metrics = scraper.get_final_snapshot()
        scraper.stop()
        if final_metrics:
            record = {"timestamp": time.time(), "final": True, **final_metrics}
            with open(args.metrics_log, "a") as f:
                f.write(json.dumps(record) + "\n")
        logger.info(f"Metrics log saved to {args.metrics_log}")

    monitor.flush()

    logger.info(f"Saving summary to {args.output}")
    summary_df = manager.summary(0, time.time())
    summary_df.to_csv(args.output, index=False)

    logger.info(f"Monitor log saved to {args.monitor_log}")

    # Auto-generate plots
    if not args.no_plot:
        try:
            from plot_monitor import plot_all
            output_dir = os.path.dirname(args.output) or "."
            plot_all(
                monitor_path=args.monitor_log,
                summary_path=args.output,
                output_dir=output_dir,
                gap_mean=args.gap_mean,
                gap_std=args.gap_std,
                output_lambda=args.output_lambda,
                metrics_path=getattr(args, 'metrics_log', None),
            )
            logger.info(f"Plots saved to {output_dir}/experiment_monitor.png")
        except ImportError:
            logger.warning("plot_monitor.py not found, skipping plot generation")
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")


if __name__ == "__main__":
    main()
