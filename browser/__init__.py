"""Read-only Tk browsers for filesystems and reference tables."""

from .filesystem import browse_filesystem
from .table import browse_table
from .viewers import register_viewer

__all__ = ["browse_filesystem", "browse_table", "register_viewer"]
