"""Mount physical files as infile filesystems and explore them with debug logging."""

import json
from pathlib import Path

from lazyfields.readers import debug_readers
from lazyfields.filesystem import FileSystem
from lazyfields.filesystem.explorer import explore


def main() -> None:
    debug_readers()
    root = Path(__file__).resolve().parent / "data"
    fs = FileSystem.from_directory(root, "**/*.json")
    spec_path = Path(__file__).with_suffix(".json")
    spec_path.write_text(json.dumps(fs.to_spec(), indent=2) + "\n", encoding="utf-8")
    explore(fs)


if __name__ == "__main__":
    main()
