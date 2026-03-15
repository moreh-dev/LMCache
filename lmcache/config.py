# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from lmcache.v1.metadata import LMCacheMetadata
from typing import Tuple

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class LMCacheMemPoolMetadata:
    """Subset of `LMCacheMetadata` to initialize MemPool"""

    kv_shape: Tuple[int, int, int, int, int]
    kv_dtype: torch.dtype
    max_local_cache_size: int


blend_default_separator = "[BLEND_SEP]"



class LMCacheEngineMetadata(LMCacheMetadata):
    """Compatibility shim for vLLM adapters expecting the older metadata name."""

    def __init__(
        self,
        model: str,
        world_size: int,
        rank: int,
        fmt: str,
        kv_dtype,
        kv_shape,
        use_mla: bool,
    ):
        super().__init__(
            model_name=model,
            world_size=world_size,
            local_world_size=world_size,
            worker_id=rank,
            local_worker_id=rank,
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
            use_mla=use_mla,
        )
        self.fmt = fmt
