#!/usr/bin/env python3
"""Patch vLLM v0.17.x scheduler for hybrid external-KV validation.

This removes the hard assert in
`vllm/v1/core/sched/scheduler.py::_mamba_block_aligned_split()` that rejects
`num_external_computed_tokens > 0` for hybrid Mamba+attention models.

Use this only when validating LMCache with hybrid-architecture models such as
Qwen3.5 on vLLM v1. It patches the installed vLLM package in-place.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


SCHEDULER_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
)


OLD = """        assert num_external_computed_tokens == 0, (
            "External KV connector is not verified yet"
        )
"""


NEW = """        # Hybrid external-KV patch: allow scheduler split path to account
        # for externally computed tokens on Mamba+attention models.
        # assert num_external_computed_tokens == 0, (
        #     "External KV connector is not verified yet"
        # )
        pass
"""


def main() -> int:
    if not SCHEDULER_PATH.exists():
        print(f"ERROR: scheduler.py not found at {SCHEDULER_PATH}")
        return 1

    content = SCHEDULER_PATH.read_text(encoding="utf-8")
    if "Hybrid external-KV patch" in content:
        print("Already patched")
        return 0

    if OLD not in content:
        print("ERROR: target assert block not found; unexpected vLLM version")
        return 1

    backup = SCHEDULER_PATH.with_suffix(SCHEDULER_PATH.suffix + ".bak_extkv")
    shutil.copy2(SCHEDULER_PATH, backup)
    SCHEDULER_PATH.write_text(content.replace(OLD, NEW, 1), encoding="utf-8")

    print(f"PATCHED: {SCHEDULER_PATH}")
    print(f"Backup:  {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
