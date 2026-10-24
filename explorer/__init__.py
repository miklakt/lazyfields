"""Read-only Tk explorers for filesystems and reference tables."""

from .browser import explore
from .table import explore_table

__all__ = ["explore", "explore_table"]
