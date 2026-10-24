"""Expose partitions at root with referenced metadata and merged block files."""

import json
from pathlib import Path

from lazyfields.readers import debug_readers
from lazyfields.filesystem import FileSystem
from lazyfields.filesystem.merge import collect, prefer, prefix, stack
from lazyfields.filesystem.explorer import explore


def main() -> None:
    debug_readers()
    root = Path(__file__).resolve().parent / "data"
    source = Path("partitions.json")
    fs = FileSystem(source, relative_to=root)
    view = fs.clone(f"{source}/partitions")
    # Explicit discovery: read the stored references once, then bind lazy targets.
    for name, partition in fs.read(f"{source}/partitions").items():
        target = FileSystem.from_file(root / source.parent / partition["file_ref"])
        view.replace(f"{name}/file_ref", target.node())
        sources = {
            Path(reference).stem: FileSystem.from_file(root / source.parent / reference).node()
            for reference in partition["file_refs"]
        }
        blocks = FileSystem.merge(sources, rules={
            "description": prefix(),
            "block_index": collect(),
            "coordinates": prefer(*sources),
            "values": stack(axis=0),
        })
        view.replace(f"{name}/file_refs", blocks.node())
    spec_path = Path(__file__).with_suffix(".json")
    spec_path.write_text(json.dumps(view.to_spec(), indent=2) + "\n", encoding="utf-8")
    explore(view)


if __name__ == "__main__":
    main()
