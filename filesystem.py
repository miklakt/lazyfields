"""Path indexing and lazy handles; physical format operations live in Reader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from .assets import AssetsCollection


@dataclass(frozen=True)
class PhysicalLocator:
    """A file/path/selection triple with an explicit physical file boundary."""

    file_path: Path
    infile_path: str | None = None
    selection: Any | None = None


@dataclass(frozen=True)
class AssetNode:
    """Lazy handle. Infile existence and type are checked only on access."""

    _filesystem: AssetsCollection = field(repr=False)
    path: str

    def read(self, selection: Any | None = None) -> Any:
        return self._filesystem.read(self.path, selection)

    def listdir(self) -> tuple[str, ...]:
        return self._filesystem.listdir(self.path)

    @property
    def locator(self) -> PhysicalLocator:
        return self._filesystem.locate(self.path)

    @property
    def kind(self) -> str:
        """Use reader metadata when available; fallback readers may load data."""
        return self._filesystem._info(self.path, list_children=False).kind


class _MountIndex:
    """Index only registered files and directories, never infile nodes."""

    def __init__(self, paths: Iterable[Path]):
        self.files = {path.as_posix(): path for path in paths}
        self.directories: dict[str, dict[str, None]] = {"": {}}
        for path in self.files:
            parent = ""
            for name in path.split("/"):
                if parent in self.files:
                    raise ValueError(f"A file mount cannot contain another file mount: {path}")
                self.directories.setdefault(parent, {})[name] = None
                parent = f"{parent}/{name}" if parent else name

    @staticmethod
    def normalize(path: str | Path) -> str:
        if not isinstance(path, (str, Path)):
            raise TypeError("Filesystem paths must be strings or Path objects.")
        parts = [part for part in str(path).split("/") if part not in ("", ".")]
        if ".." in parts:
            raise ValueError("Filesystem paths cannot contain '..'.")
        return "/".join(parts)

    def split(self, path: str) -> tuple[Path | None, str | None]:
        if path in self.directories:
            return None, None
        prefix = ""
        for name in path.split("/"):
            prefix = f"{prefix}/{name}" if prefix else name
            if prefix in self.files:
                return self.files[prefix], path[len(prefix) + 1:] or None
        raise KeyError(f"Unknown filesystem path: {path!r}")
