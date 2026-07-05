from unittest.mock import MagicMock, patch

from hierocode.engine import draft_unit
from hierocode.config import HierocodeConfig

def test_engine_happy_path(tmp_path):
    (tmp_path / "target.py").write_text("def foo():\n    pass\n")
    
    config = MagicMock(spec=HierocodeConfig)
    config.providers = {"drafter_provider": MagicMock()}
    
    with patch("hierocode.engine.get_route", return_value=("drafter_provider", "model")):
        with patch("hierocode.engine.get_provider") as mock_get_provider:
            provider = MagicMock()
            provider.generate.return_value = "def foo():\n    return 1\n"
            mock_get_provider.return_value = provider
            
            with patch("hierocode.engine.build_capacity_profile") as mock_build_profile:
                profile = MagicMock()
                profile.max_input_tokens = 1000
                profile.max_output_tokens = 1000
                profile.max_files_per_unit = 10
                mock_build_profile.return_value = profile
                
                result = draft_unit(
                    goal="g",
                    target_file="target.py",
                    repo_root=tmp_path,
                    config=config
                )
                
                assert result.status == "ok"
                assert "+    return 1" in result.diff

def test_engine_budget_error(tmp_path):
    (tmp_path / "target.py").write_text("x\n" * 10000)
    
    config = MagicMock(spec=HierocodeConfig)
    config.providers = {"drafter_provider": MagicMock()}
    
    with patch("hierocode.engine.get_route", return_value=("drafter_provider", "model")):
        with patch("hierocode.engine.get_provider") as mock_get_provider:
            provider = MagicMock()
            mock_get_provider.return_value = provider
            
            with patch("hierocode.engine.build_capacity_profile") as mock_build_profile:
                profile = MagicMock()
                profile.max_input_tokens = 10  # Very small!
                profile.max_output_tokens = 10
                profile.max_files_per_unit = 10
                mock_build_profile.return_value = profile
                
                result = draft_unit(
                    goal="g",
                    target_file="target.py",
                    repo_root=tmp_path,
                    config=config
                )
                
                assert result.status == "error"
                assert result.error_type == "budget"
                assert "target file(s)" in result.error_message

def test_engine_empty_diff(tmp_path):
    (tmp_path / "target.py").write_text("def foo():\n    pass\n")
    
    config = MagicMock(spec=HierocodeConfig)
    config.providers = {"drafter_provider": MagicMock()}
    
    with patch("hierocode.engine.get_route", return_value=("drafter_provider", "model")):
        with patch("hierocode.engine.get_provider") as mock_get_provider:
            provider = MagicMock()
            # Return same content
            provider.generate.return_value = "def foo():\n    pass\n"
            mock_get_provider.return_value = provider
            
            with patch("hierocode.engine.build_capacity_profile") as mock_build_profile:
                profile = MagicMock()
                profile.max_input_tokens = 1000
                profile.max_output_tokens = 1000
                profile.max_files_per_unit = 10
                mock_build_profile.return_value = profile
                
                result = draft_unit(
                    goal="g",
                    target_file="target.py",
                    repo_root=tmp_path,
                    config=config
                )

                assert result.status == "error"
                assert result.error_type == "empty"

def test_engine_edit_apply_failure_reported(tmp_path):
    """Malformed edit blocks on both attempts → error_type='edit_apply', not 'empty'."""
    (tmp_path / "target.py").write_text("def foo():\n    pass\n")

    config = MagicMock(spec=HierocodeConfig)
    config.providers = {"drafter_provider": MagicMock()}

    # Missing '=======' divider — parse_edit_blocks raises EditApplyError every time.
    malformed = "<<<<<<< SEARCH\ndef foo():\n>>>>>>> REPLACE"

    with patch("hierocode.engine.get_route", return_value=("drafter_provider", "model")):
        with patch("hierocode.engine.get_provider") as mock_get_provider:
            provider = MagicMock()
            provider.generate.return_value = malformed
            mock_get_provider.return_value = provider

            with patch("hierocode.engine.build_capacity_profile") as mock_build_profile:
                profile = MagicMock()
                profile.max_input_tokens = 1000
                profile.max_output_tokens = 1000
                profile.max_files_per_unit = 10
                mock_build_profile.return_value = profile

                result = draft_unit(
                    goal="g",
                    target_file="target.py",
                    repo_root=tmp_path,
                    config=config
                )

                assert result.status == "error"
                assert result.error_type == "edit_apply"
                assert "could not be applied" in result.error_message
                assert provider.generate.call_count == 2  # one retry, then give up
