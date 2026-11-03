"""Read-only Tk browsers for filesystems and reference tables."""

from .methods import DEFAULT_METHODS, Method, Methods
from .filesystem import browse_filesystem
from .table import browse_table
from .extra_methods import ARRAY_ACTIONS, EXTRA_VIEWERS, FIGURE, IMAGE, PLOT, SCATTER, TABLE

__all__ = ["browse_filesystem", "browse_table", "Method", "Methods", "DEFAULT_METHODS",
           "EXTRA_VIEWERS", "ARRAY_ACTIONS", "TABLE", "PLOT", "SCATTER", "FIGURE", "IMAGE"]
