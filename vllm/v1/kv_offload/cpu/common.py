# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing_extensions import override

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


class CPUOffloadingMetrics:
    STORES_SKIPPED = "vllm:kv_offload_stores_skipped"
    CPU_CACHE_USAGE_PERC = "vllm:kv_offload_cpu_cache_usage_perc"
    CPU_BLOCK_LOOKUP = "vllm:kv_offload_cpu_block_lookup_total"
    CPU_BLOCK_HIT = "vllm:kv_offload_cpu_block_hit_total"
    CPU_BLOCK_MISS = "vllm:kv_offload_cpu_block_miss_total"
    BLOCK_EVICTION = "vllm:kv_offload_block_eviction_total"


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.
    """

    @staticmethod
    @override
    def medium() -> str:
        return "CPU"
