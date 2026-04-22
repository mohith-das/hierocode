"""Build a CapacityProfile describing what the drafter model can handle."""

import re
from typing import Optional

from hierocode.broker.plan_schema import CapacityProfile
from hierocode.providers.base import BaseProvider
from hierocode.runtime.gpu import probe_gpu
from hierocode.runtime.resources import get_cpu_count, get_total_ram_gb

_TIER_TABLE = [
    (2.0, "micro", 500, 1),
    (5.0, "narrow", 1500, 1),
    (10.0, "standard", 3000, 2),
    (20.0, "capable", 5000, 3),
]
_STRONG = ("strong", 8000, 5)
_NARROW_UNKNOWN = ("narrow", 1500, 1)

_PARAM_RE = re.compile(r"[:_-](\d+(?:\.\d+)?)[bB]")


def _tier_for(param_count_b: Optional[float]) -> tuple[str, int, int]:
    """Return (tier, max_output_tokens, max_files_per_unit) for a param count."""
    if param_count_b is None:
        return _NARROW_UNKNOWN
    for threshold, tier, max_out, max_files in _TIER_TABLE:
        if param_count_b < threshold:
            return tier, max_out, max_files
    return _STRONG


def _parse_param_count(model: str) -> Optional[float]:
    """Extract param count in billions from a model name string, or None."""
    match = _PARAM_RE.search(model)
    if match:
        return float(match.group(1))
    return None


def _compute_budget(num_ctx: int, max_output_tokens: int) -> int:
    """Compute max_input_tokens with a 20% safety margin; clamp if needed."""
    safety_margin = int(num_ctx * 0.2)
    prompt_overhead = 500
    raw = num_ctx - safety_margin - max_output_tokens - prompt_overhead
    if raw >= 512:
        return raw
    return max(256, num_ctx // 2)


def build_capacity_profile(
    drafter_provider: BaseProvider,
    drafter_model: str,
    resource_overrides: Optional[dict] = None,
) -> CapacityProfile:
    """Build a CapacityProfile from provider model metadata and host resources."""
    # --- model info ---
    try:
        info = drafter_provider.get_model_info(drafter_model)  # type: ignore[attr-defined]
        num_ctx: int = info["num_ctx"]
        param_count_b: Optional[float] = info.get("param_count_b")
        quantization: Optional[str] = info.get("quantization")
    except Exception:
        num_ctx = 8192
        param_count_b = _parse_param_count(drafter_model)
        quantization = None

    tier, max_output_tokens, max_files_per_unit = _tier_for(param_count_b)
    max_input_tokens = _compute_budget(num_ctx, max_output_tokens)

    # --- host resources ---
    if resource_overrides is not None:
        host_ram_gb: float = resource_overrides["ram_gb"]
        host_cpu_cores: int = resource_overrides["cpu_cores"]
        host_vram_gb: float = resource_overrides.get("vram_gb", 0.0)
        has_gpu: bool = resource_overrides.get("has_gpu", False)
    else:
        host_ram_gb = get_total_ram_gb()
        host_cpu_cores = get_cpu_count()
        gpu = probe_gpu()
        host_vram_gb = gpu.vram_gb
        has_gpu = gpu.has_gpu

    return CapacityProfile(
        drafter_model=drafter_model,
        param_count_b=param_count_b,
        quantization=quantization,
        context_window=num_ctx,
        host_ram_gb=host_ram_gb,
        host_vram_gb=host_vram_gb,
        host_cpu_cores=host_cpu_cores,
        has_gpu=has_gpu,
        tier=tier,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_files_per_unit=max_files_per_unit,
    )
