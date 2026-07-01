# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

logger = init_logger(__name__)

_SUPPORTED_POLICIES = ("lru", "arc", "sae")

_SAE_TUNABLE_DEFAULTS: dict[str, tuple[type, object]] = {
    "sae_decay_interval": (int, 500),
    "sae_decay_factor": (float, 0.9),
    "sae_ghost_hit_weight": (float, 12.0),
    "sae_ghost_miss_weight": (float, 1.0),
    "sae_ghost_norm": (float, 12.0),
}


def _validate_sae_tunables(extra_config: dict[str, Any]) -> dict[str, Any]:
    """Extract and validate SAE tunables from extra_config.

    Returns:
        A dict of ``SAECachePolicy`` constructor kwargs
        (``decay_interval``, ``decay_factor``, ``ghost_hit_weight``,
        ``ghost_miss_weight``, ``ghost_norm``).

    Raises:
        ValueError: on out-of-range values, naming the offending key.
    """
    kwargs: dict[str, Any] = {}
    for key, (expected_type, default) in _SAE_TUNABLE_DEFAULTS.items():
        raw = extra_config.get(key, default)
        try:
            value = expected_type(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{key}={raw!r} is not a valid {expected_type.__name__}"
            ) from exc

        if key == "sae_decay_interval" and value < 1:
            raise ValueError(f"{key}={value} must be >= 1")
        if key == "sae_decay_factor" and not (0.0 < value <= 1.0):
            raise ValueError(f"{key}={value} must satisfy 0.0 < x <= 1.0")
        if key == "sae_ghost_hit_weight" and value < 0.0:
            raise ValueError(f"{key}={value} must be >= 0.0")
        if key == "sae_ghost_miss_weight" and value < 0.0:
            raise ValueError(f"{key}={value} must be >= 0.0")
        if key == "sae_ghost_norm" and value <= 0.0:
            raise ValueError(f"{key}={value} must be > 0.0")

        # Strip "sae_" prefix for the CachePolicy constructor kwarg name.
        kwargs[key[len("sae_") :]] = value
    return kwargs


class CPUOffloadingSpec(OffloadingSpec):
    BLOCK_SIZE_ALIGNMENT = 1

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        definitions: dict[str, OffloadingMetricMetadata] = {
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by active "
                    "transfers (0.0 = idle, 1.0 = saturated). Sustained high "
                    "values indicate transfers (stores or promotions) may be "
                    "dropped due to insufficient capacity."
                ),
            ),
            CPUOffloadingMetrics.CPU_BLOCK_LOOKUP: OffloadingCounterMetadata(
                documentation=(
                    "Total CPU KV cache lookup calls, labelled by eviction "
                    "policy (lru/arc/sae). Sum of hits and misses "
                    "(HIT_PENDING counts as a hit; RETRY is not counted)."
                ),
                labelnames=("policy",),
            ),
            CPUOffloadingMetrics.CPU_BLOCK_HIT: OffloadingCounterMetadata(
                documentation=(
                    "Total CPU KV cache lookup hits, labelled by eviction "
                    "policy (lru/arc/sae). HIT_PENDING counts as a hit."
                ),
                labelnames=("policy",),
            ),
            CPUOffloadingMetrics.CPU_BLOCK_MISS: OffloadingCounterMetadata(
                documentation=(
                    "Total CPU KV cache lookup misses, labelled by eviction "
                    "policy (lru/arc/sae)."
                ),
                labelnames=("policy",),
            ),
            CPUOffloadingMetrics.BLOCK_EVICTION: OffloadingCounterMetadata(
                documentation=(
                    "Total CPU KV cache blocks evicted, labelled by eviction "
                    "policy (lru/arc/sae)."
                ),
                labelnames=("policy",),
            ),
        }
        store_threshold = int(extra_config.get("store_threshold", 0))
        if store_threshold >= 2:
            definitions[CPUOffloadingMetrics.STORES_SKIPPED] = (
                OffloadingCounterMetadata(
                    documentation=(
                        "Number of KV offload stores skipped because the reuse "
                        "threshold was not reached."
                    ),
                )
            )
        return definitions

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        world_size = vllm_config.parallel_config.world_size
        self.num_blocks = 0
        self.kv_bytes_per_offloaded_block = 0
        self.cpu_page_size_per_worker = 0
        assert kv_cache_config is not None
        if kv_cache_config.num_blocks > 0 and world_size > 0:
            is_packed = any(t.block_stride for t in kv_cache_config.kv_cache_tensors)
            assert not is_packed or all(
                t.block_stride for t in kv_cache_config.kv_cache_tensors
            )
            total_gpu_kv_bytes = (
                kv_cache_config.kv_cache_tensors[0].size
                if is_packed
                else sum(t.size for t in kv_cache_config.kv_cache_tensors)
            )
            kv_bytes_per_block = (
                total_gpu_kv_bytes // kv_cache_config.num_blocks
            ) * world_size
            kv_bytes_per_offloaded_block = kv_bytes_per_block * self.block_size_factor

            # calculate cpu_page_size_per_worker
            self.cpu_page_size_per_worker = kv_bytes_per_offloaded_block // world_size

            # calculate num_blocks
            aligned_kv_bytes_per_offloaded_block = round_up(
                kv_bytes_per_offloaded_block, self.BLOCK_SIZE_ALIGNMENT
            )
            self.num_blocks = (
                int(cpu_bytes_to_use) // aligned_kv_bytes_per_offloaded_block
            )

            # Expose aligned_kv_bytes_per_offloaded_block as
            # kv_bytes_per_offloaded_block. Note that this might contain
            # some padding. i.e. each offloaded block is of the form,
            # |--- W0-B0---|---- W1-B0---| ... |---- Wn-B0---| *** maybe-pad *** |
            self.kv_bytes_per_offloaded_block = aligned_kv_bytes_per_offloaded_block

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._worker: CPUOffloadingWorker | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        if self.eviction_policy not in _SUPPORTED_POLICIES:
            raise ValueError(
                f"eviction_policy={self.eviction_policy!r} is not supported. "
                f"Supported: {list(_SUPPORTED_POLICIES)}"
            )

        offending_sae_keys = [k for k in self.extra_config if k.startswith("sae_")]
        if self.eviction_policy != "sae" and offending_sae_keys:
            raise ValueError(
                f"SAE-specific keys {offending_sae_keys!r} are set but "
                f"eviction_policy={self.eviction_policy!r} is not 'sae'."
            )

        self._sae_policy_kwargs: dict[str, Any] = (
            _validate_sae_tunables(self.extra_config)
            if self.eviction_policy == "sae"
            else {}
        )

        logger.info("CPU offload: eviction_policy=%s", self.eviction_policy)

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            # store_threshold: how many times a block must appear in lookup()
            # before it is eligible for CPU offloading.  Values < 2 disable
            # filtering (a threshold of 1 equals no filter; 0 is the default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
                policy_kwargs=self._sae_policy_kwargs,
            )
        return self._manager

    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        return CPUOffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
        )

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike "
                    "and XPU GPUs"
                )
            self._worker = self.create_worker(kv_caches)

        assert self._worker is not None
        return self._worker
