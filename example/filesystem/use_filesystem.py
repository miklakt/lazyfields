"""Explore the registered fixture files."""

from pathlib import Path

from lazyfields.filesystem import FileSystem
from explorer import explore


def main() -> None:
    root = Path(__file__).resolve().parent / "data"
    fs = FileSystem(relative_to=root)
    for path in sorted(root.rglob("*.json")):
        fs.mount(path.relative_to(root), FileSystem.from_file(path).node())
    explore(fs)


if __name__ == "__main__":
    main()
