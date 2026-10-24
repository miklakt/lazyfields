"""Read-only Tk browsers for filesystems and reference tables."""

from .filesystem import browse_filesystem
from .table import browse_table

__all__ = ["browse_filesystem", "browse_table"]
