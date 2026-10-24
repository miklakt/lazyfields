"""Expose partitions at root with referenced metadata and merged block files."""

import json
import sys
from pathlib import Path

import pandas as pd

import lazyfields as lf
from lazyfields.readers import debug_readers
from lazyfields.filesystem import FileSystem
from lazyfields.filesystem.merge import collect, prefer, prefix, stack
from lazyfields.browser import browse_table


def compose(root: Path) -> FileSystem:
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
    return view


def add_to_index(path):
    """Pipe step: copy a scalar buried at path into a column named after its last key."""
    def step(row) -> None:
        row[path.rsplit("/", 1)[-1]] = row.store[path]
    return step


def main() -> None:
    debug_readers()
    root = Path(__file__).resolve().parent / "data"
    spec_path = Path(__file__).with_suffix(".json")
    reference_path = Path(__file__).with_suffix(".pkl")
    rebuild = "--rebuild" in sys.argv
    # Reuse the saved specification; pass --rebuild to compose and save it again.
    if spec_path.exists() and not rebuild:
        view = FileSystem.from_spec(json.loads(spec_path.read_text(encoding="utf-8")), relative_to=root)
    else:
        view = compose(root)
        spec_path.write_text(json.dumps(view.to_spec(), indent=2) + "\n", encoding="utf-8")
    # One row per partition at the view root; arrays stay lazy behind each row.
    if reference_path.exists() and not rebuild:
        reference_df = pd.read_pickle(reference_path)
    else:
        # file_ref/label lands in row["label"], file_ref/weight in row["weight"].
        pipe = [add_to_index("file_ref/label"), add_to_index("file_ref/weight")]
        reference_df = lf.create_reference_table(view.to_spec(), relative_to=root, pipe=pipe)
        reference_df.to_pickle(reference_path)
    browse_table(reference_df)


if __name__ == "__main__":
    main()
