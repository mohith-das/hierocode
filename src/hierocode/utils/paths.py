import os
from pathlib import Path

def get_config_path() -> Path:
    """Returns the path to the user's Hierocode config file."""
    home_dir = Path(os.path.expanduser("~"))
    return home_dir / ".hierocode.yaml"
