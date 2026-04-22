from hierocode.models.schemas import HierocodeConfig, PolicyConfig


def test_config_parsing_defaults():
    config = HierocodeConfig()
    assert config.default_provider == "local_default"
    assert config.parallelization.default_strategy == "balanced"
    assert config.routing.planner is None
    assert config.routing.drafter is None
    assert config.policy.max_revisions_per_unit == 2


def test_config_parsing_custom():
    data = {"default_provider": "remote_one"}
    config = HierocodeConfig(**data)
    assert config.default_provider == "remote_one"


def test_config_ignores_legacy_fields():
    """v0.1 configs with default_model / small_model / routing.draft_model parse without error."""
    data = {
        "default_provider": "p",
        "default_model": "ignored",
        "small_model": "ignored",
        "routing": {"draft_model": "ignored", "review_model": "ignored"},
    }
    config = HierocodeConfig(**data)
    assert config.default_provider == "p"
    assert config.routing.planner is None


def test_policy_config_default_auto_apply_false():
    """PolicyConfig.auto_apply defaults to False."""
    policy = PolicyConfig()
    assert policy.auto_apply is False


def test_policy_config_accepts_auto_apply_true():
    """PolicyConfig.auto_apply can be set to True."""
    policy = PolicyConfig(auto_apply=True)
    assert policy.auto_apply is True


def test_hieroconfig_default_policy_auto_apply_false():
    """HierocodeConfig default policy has auto_apply=False."""
    config = HierocodeConfig()
    assert config.policy.auto_apply is False
