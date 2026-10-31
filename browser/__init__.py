"""Read-only Tk browsers for filesystems and reference tables."""

from .filesystem import browse_filesystem
from .table import browse_table
from .methods import Method, register_method

__all__ = ["browse_filesystem", "browse_table", "register_method", "Method"]
