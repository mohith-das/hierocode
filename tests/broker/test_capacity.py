"""Tests for hierocode.broker.capacity."""

import pytest
from unittest.mock import MagicMock, patch

from hierocode.broker.plan_schema import CapacityProfile
from hierocode.providers.base import BaseProvider
from hierocode.broker.capacity import build_capacity_profile

_OVERRIDES = {"ram_gb": 16.0, "cpu_cores": 8}


def _make_provider(info: dict) -> MagicMock:
    provider = MagicMock(spec=BaseProvider)
    provider.get_model_info = MagicMock(return_value=info)
    return provider


def _make_raising_provider() -> MagicMock:
    provider = MagicMock(spec=BaseProvider)
    provider.get_model_info = MagicMock(side_effect=RuntimeError("unavailable"))
    return provider


class TestUsesGetModelInfoValues:
    def test_uses_get_model_info_values(self):
        info = {"num_ctx": 16384, "param_count_b": 8.0, "quantization": "Q4_K_M"}
        provider = _make_provider(info)
        profile = build_capacity_profile(provider, "mymodel:8b", resource_overrides=_OVERRIDES)
        assert profile.context_window == 16384
        assert profile.param_count_b == 8.0
        assert profile.quantization == "Q4_K_M"
        assert profile.tier == "standard"


class TestFallbackOnGetModelInfoException:
    def test_falls_back_on_get_model_info_exception(self):
        provider = _make_raising_provider()
        profile = build_capacity_profile(provider, "llama3.2:3b", resource_overrides=_OVERRIDES)
        assert profile.context_window == 8192
        assert profile.param_count_b == 3.0
        assert profile.quantization is None
        assert profile.tier == "narrow"


class TestTierBoundaries:
    @pytest.mark.parametrize(
        "param_count_b, expected_tier",
        [
            (1.0, "micro"),
            (2.0, "narrow"),
            (3.0, "narrow"),
            (5.0, "standard"),
            (8.0, "standard"),
            (13.0, "capable"),
            (20.0, "strong"),
            (70.0, "strong"),
        ],
    )
    def test_tier_boundaries(self, param_count_b: float, expected_tier: str):
        info = {"num_ctx": 8192, "param_count_b": param_count_b, "quantization": None}
        provider = _make_provider(info)
        profile = build_capacity_profile(provider, "model", resource_overrides=_OVERRIDES)
        assert profile.tier == expected_tier


class TestUnknownParamCount:
    def test_unknown_param_count_is_narrow(self):
        info = {"num_ctx": 8192, "param_count_b": None, "quantization": None}
        provider = _make_provider(info)
        profile = build_capacity_profile(provider, "unknown-model", resource_overrides=_OVERRIDES)
        assert profile.tier == "narrow"
        assert profile.param_count_b is None


class TestBudgetRespectedNumCtx:
    def test_budget_respects_num_ctx(self):
        # param_count_b=3.0 → narrow → max_output=1500
        # safety_margin = int(4096 * 0.2) = 819
        # max_input = 4096 - 819 - 1500 - 500 = 1277
        info = {"num_ctx": 4096, "param_count_b": 3.0, "quantization": None}
        provider = _make_provider(info)
        profile = build_capacity_profile(provider, "model:3b", resource_overrides=_OVERRIDES)
        assert profile.max_output_tokens == 1500
        assert profile.max_input_tokens >= 512
        assert profile.max_input_tokens <= 4096 - int(4096 * 0.2) - 1500 - 500


class TestBudgetClampTinyContext:
    def test_budget_clamp_for_tiny_context(self):
        # num_ctx=1024, param_count_b=3.0 → narrow → max_output=1500
        # raw = 1024 - 204 - 1500 - 500 = -1180 → clamp to max(256, 512) = 512
        info = {"num_ctx": 1024, "param_count_b": 3.0, "quantization": None}
        provider = _make_provider(info)
        profile = build_capacity_profile(provider, "model:3b", resource_overrides=_OVERRIDES)
        assert isinstance(profile, CapacityProfile)
        assert profile.max_input_tokens >= 256


class TestResourceOverridesApplied:
    def test_resource_overrides_applied(self):
        info = {"num_ctx": 8192, "param_count_b": 8.0, "quantization": None}
        provider = _make_provider(info)
        overrides = {"ram_gb": 64.0, "cpu_cores": 16, "vram_gb": 24.0, "has_gpu": True}
        profile = build_capacity_profile(provider, "model:8b", resource_overrides=overrides)
        assert profile.host_ram_gb == 64.0
        assert profile.host_cpu_cores == 16
        assert profile.host_vram_gb == 24.0
        assert profile.has_gpu is True


class TestUsesGetTotalRamGbWhenNoOverrides:
    def test_uses_get_total_ram_gb_when_no_overrides(self):
        info = {"num_ctx": 8192, "param_count_b": 8.0, "quantization": None}
        provider = _make_provider(info)
        with patch("hierocode.broker.capacity.get_total_ram_gb", return_value=32.0):
            profile = build_capacity_profile(provider, "model:8b")
        assert profile.host_ram_gb == 32.0
