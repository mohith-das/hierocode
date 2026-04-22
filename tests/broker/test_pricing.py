"""Tests for hierocode.broker.pricing — loader, merge logic, cache, and path resolution."""

from pathlib import Path
from unittest.mock import patch

import pytest

import hierocode.broker.pricing as pricing_mod
from hierocode.broker.pricing import (
    SubscriptionQuota,
    default_pricing,
    get_pricing,
    load_pricing,
    pricing_config_path,
)


# ---------------------------------------------------------------------------
# default_pricing
# ---------------------------------------------------------------------------


def test_default_pricing_returns_hardcoded():
    cfg = default_pricing()
    assert "claude-haiku-4-5" in cfg.anthropic_models
    assert "claude-sonnet-4-6" in cfg.anthropic_models
    assert "claude-opus-4-7" in cfg.anthropic_models
    assert "claude_pro" in cfg.subscription_quotas


def test_default_pricing_returns_fresh_copies():
    cfg1 = default_pricing()
    cfg2 = default_pricing()
    cfg1.anthropic_models["injected-model"] = (99.0, 99.0)
    assert "injected-model" not in cfg2.anthropic_models
    cfg1.subscription_quotas["injected_quota"] = SubscriptionQuota(1, 1)
    assert "injected_quota" not in cfg2.subscription_quotas


# ---------------------------------------------------------------------------
# load_pricing — missing / empty / malformed files
# ---------------------------------------------------------------------------


def test_load_pricing_missing_file_returns_defaults(tmp_path: Path):
    cfg = load_pricing(override_path=tmp_path / "nope.yaml")
    expected = default_pricing()
    assert cfg.anthropic_models == expected.anthropic_models
    assert cfg.subscription_quotas.keys() == expected.subscription_quotas.keys()


def test_load_pricing_empty_file_returns_defaults(tmp_path: Path):
    p = tmp_path / "pricing.yaml"
    p.write_text("", encoding="utf-8")
    cfg = load_pricing(override_path=p)
    expected = default_pricing()
    assert cfg.anthropic_models == expected.anthropic_models


def test_load_pricing_malformed_yaml_warns_and_defaults(tmp_path: Path):
    p = tmp_path / "pricing.yaml"
    p.write_text("{not valid yaml", encoding="utf-8")
    with pytest.warns(RuntimeWarning):
        cfg = load_pricing(override_path=p)
    expected = default_pricing()
    assert cfg.anthropic_models == expected.anthropic_models


def test_load_pricing_non_dict_top_level_warns_and_defaults(tmp_path: Path):
    p = tmp_path / "pricing.yaml"
    p.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.warns(RuntimeWarning):
        cfg = load_pricing(override_path=p)
    expected = default_pricing()
    assert cfg.anthropic_models == expected.anthropic_models


# ---------------------------------------------------------------------------
# load_pricing — user overrides and merges
# ---------------------------------------------------------------------------


def test_load_pricing_user_overrides_win(tmp_path: Path):
    p = tmp_path / "pricing.yaml"
    p.write_text("anthropic_models:\n  claude-haiku-4-5: [0.1, 0.5]\n", encoding="utf-8")
    cfg = load_pricing(override_path=p)
    assert cfg.anthropic_models["claude-haiku-4-5"] == (0.1, 0.5)
    # Other defaults untouched
    assert cfg.anthropic_models["claude-sonnet-4-6"] == (3.0, 15.0)
    assert cfg.anthropic_models["claude-opus-4-7"] == (15.0, 75.0)


def test_load_pricing_adds_new_anthropic_model(tmp_path: Path):
    p = tmp_path / "pricing.yaml"
    p.write_text("anthropic_models:\n  claude-future: [1.0, 4.0]\n", encoding="utf-8")
    cfg = load_pricing(override_path=p)
    assert cfg.anthropic_models["claude-future"] == (1.0, 4.0)
    # Original defaults still present
    assert "claude-haiku-4-5" in cfg.anthropic_models
    assert "claude-sonnet-4-6" in cfg.anthropic_models
    assert "claude-opus-4-7" in cfg.anthropic_models


def test_load_pricing_skips_invalid_anthropic_entry(tmp_path: Path):
    p = tmp_path / "pricing.yaml"
    p.write_text("anthropic_models:\n  claude-bad: \"not a list\"\n", encoding="utf-8")
    with pytest.warns(RuntimeWarning):
        cfg = load_pricing(override_path=p)
    assert "claude-bad" not in cfg.anthropic_models
    # Defaults intact
    assert "claude-haiku-4-5" in cfg.anthropic_models


def test_load_pricing_skips_invalid_price_pair(tmp_path: Path):
    p = tmp_path / "pricing.yaml"
    p.write_text("anthropic_models:\n  claude-bad: [1.0]\n", encoding="utf-8")
    with pytest.warns(RuntimeWarning):
        cfg = load_pricing(override_path=p)
    assert "claude-bad" not in cfg.anthropic_models


def test_load_pricing_subscription_override(tmp_path: Path):
    p = tmp_path / "pricing.yaml"
    p.write_text(
        "subscription_quotas:\n  claude_pro:\n    messages_per_window: 80\n    window_hours: 5\n",
        encoding="utf-8",
    )
    cfg = load_pricing(override_path=p)
    assert cfg.subscription_quotas["claude_pro"].messages_per_window == 80
    assert cfg.subscription_quotas["claude_pro"].window_hours == 5


def test_load_pricing_subscription_add_new_provider(tmp_path: Path):
    p = tmp_path / "pricing.yaml"
    p.write_text(
        "subscription_quotas:\n  kimi_pro:\n    messages_per_window: 100\n    window_hours: 24\n",
        encoding="utf-8",
    )
    cfg = load_pricing(override_path=p)
    assert "kimi_pro" in cfg.subscription_quotas
    assert cfg.subscription_quotas["kimi_pro"].messages_per_window == 100
    assert cfg.subscription_quotas["kimi_pro"].window_hours == 24
    # Defaults still present
    assert "claude_pro" in cfg.subscription_quotas
    assert "chatgpt_plus" in cfg.subscription_quotas


def test_load_pricing_skips_invalid_quota_entry(tmp_path: Path):
    p = tmp_path / "pricing.yaml"
    p.write_text('subscription_quotas:\n  claude_pro: "not a dict"\n', encoding="utf-8")
    with pytest.warns(RuntimeWarning):
        cfg = load_pricing(override_path=p)
    # claude_pro keeps the default (skipped override)
    assert cfg.subscription_quotas["claude_pro"].messages_per_window == 40


# ---------------------------------------------------------------------------
# get_pricing — caching
# ---------------------------------------------------------------------------


def test_get_pricing_caches():
    """Two calls without reload= return the same object; load_pricing called once."""
    # Reset the module-level cache first
    pricing_mod._CACHED = None
    with patch.object(pricing_mod, "load_pricing", wraps=pricing_mod.load_pricing) as mock_load:
        first = get_pricing()
        second = get_pricing()
    assert first is second
    mock_load.assert_called_once()


def test_get_pricing_reload_forces_reload():
    """reload=True causes load_pricing to be called again."""
    pricing_mod._CACHED = None
    with patch.object(pricing_mod, "load_pricing", wraps=pricing_mod.load_pricing) as mock_load:
        get_pricing()
        get_pricing(reload=True)
    assert mock_load.call_count == 2


# ---------------------------------------------------------------------------
# pricing_config_path
# ---------------------------------------------------------------------------


def test_pricing_config_path_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    p = pricing_config_path()
    assert p == Path.home() / ".hierocode" / "pricing.yaml"


def test_pricing_config_path_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    p = pricing_config_path()
    assert p == tmp_path / "hierocode" / "pricing.yaml"
