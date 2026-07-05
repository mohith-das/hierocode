"""
Hierocode
A local-first hierarchical coding orchestrator for delegating tasks.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hierocode")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"
