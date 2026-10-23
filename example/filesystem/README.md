# Editable filesystem example

`FileSystem.from_file(path, reader=None)` wraps a reader as a filesystem rooted
at a file's contents. Mount these trees to compose a larger filesystem, then edit
or clone any subtree. Values remain read-only in their source files. File-backed
roots, composed filesystems, and clones use the same implementation.

The filesystem does not interpret strings as links. The example chooses
which stored references to resolve and which nodes to replace.

## Run the explorers

With the checkout installed in the active Python environment and Tk available:

```bash
python3 example/filesystem/use_filesystem.py
python3 example/filesystem/use_mounts.py
```

The first explorer shows the physical filesystem. The second clones it, then
replaces the partition reference fields with the referenced files. Expand groups
on the left or double-click children on the right. Leaf values are read when
visited. OS directories show `📁`, file roots `📄`, other groups `⊞`, datasets
`▦`, and scalars `·` in both panes; only groups have expansion arrows. Children are inspected one level at a time. Returning to a parent
shows remembered short text previews; full values are not cached. Logical mount boundaries also show `↪`, so relocated file roots show `↪ 📄`.

## Fixtures

All paths are relative to `example/filesystem/data/`:

```text
partitions.json
metadata/{alpha,beta,gamma}.json
blocks/{alpha,beta,gamma}_{first,second}.json
```

`partitions.json` contains three entries under `partitions`. Each has `scalar_a`,
`scalar_b`, a length-3 `vector`, a `(2, 3)` `matrix`, a `file_ref` string, and an
ordered `file_refs` list. Metadata files contain `description`, `label`, `weight`,
and `coefficients`. Block files contain `description`, `block_index`, `coordinates`,
and `values`. The distinct fixture values help verify source order and selection.

## Files as filesystem roots

```python
from lazyfields.filesystem import FileSystem

contents = FileSystem.from_file("example/filesystem/data/metadata/alpha.json")
contents.read("coefficients", 0)  # 1.01

fs = FileSystem()
fs.mount("metadata/alpha.json", contents.node())
fs.read("metadata/alpha.json/coefficients", 0)  # 1.01
```

The reader is inferred from the filename unless supplied explicitly. Neither
`from_file` nor mounting reads file contents. `FileSystem(paths, relative_to=root)`
is shorthand for wrapping and mounting each relative filename this way. Repeated
filenames are registered once; overlapping file paths are rejected before mounting.

## Clone, unmount, mount

```python
from pathlib import Path
from lazyfields.filesystem import FileSystem

root = Path("example/filesystem/data").resolve()
fs = FileSystem(
    sorted(path.relative_to(root) for path in root.rglob("*.json")),
    relative_to=root,
)
view = fs.clone()

path = "partitions.json/partitions/alpha/file_ref"
reference = view.read(path)  # 'metadata/alpha.json'
view.unmount(path)
view.mount(path, fs.node(reference))

view.read(f"{path}/coefficients", 0)  # 1.01
fs.read(path)  # 'metadata/alpha.json'
```

`mount` requires a vacant path. Mounting over an existing node raises
`FileExistsError`; explicitly unmount it first. Missing parent groups are created,
but mounting a child beneath a scalar or dataset raises `NotADirectoryError`.
Unmounting a group removes its whole logical subtree. Physical files are unchanged.
Mounting or unmounting the root is rejected; use `clone(path)` to choose a subtree root.

Use `FileSystem()` for an empty namespace. `clone(path)` returns another
`FileSystem` rooted at that subtree. A mount target must be a node obtained
with `node(path)` from any filesystem:

```python
view = fs.clone("partitions.json/partitions/alpha")
view.unmount("file_ref")
view.mount("file_ref", fs.node("metadata/alpha.json"))

combined = FileSystem()
combined.mount("alpha", view.node())
combined.mount("metadata", fs.node("metadata"))
```

`mount` and `unmount` work on the original filesystem too. Cloning or mounting a
node copies its current namespace structure, including edits. Subsequent namespace
changes are independent. Immutable physical source bindings remain shared and values
are read afresh; dataset values are never copied. Rooted scalar and dataset clones
support the same native reads and selections as their sources.

## Reference resolution in the demo

`use_mounts.py` registers only `partitions.json`, reads the partition mapping, and resolves reference
paths relative to the parent of `partitions.json`, and calls `unmount` followed by
`mount`. Referenced files are wrapped with `FileSystem.from_file`. It mounts the
single `file_ref` target directly. It replaces `file_refs`
with a group containing the block files under ordered names `0`, `1`, etc.

```text
view/
└── partitions.json/
    └── partitions/
        └── alpha/                     beta and gamma have the same structure
            ├── scalar_a
            ├── scalar_b
            ├── vector
            ├── matrix
            ├── file_ref/              metadata/alpha.json
            │   └── coefficients
            └── file_refs/
                ├── 0/                 blocks/alpha_first.json
                │   └── values
                └── 1/                 blocks/alpha_second.json
                    └── values
```

The tree is abbreviated; mounted files retain all their fields. Raw references
remain accessible through the unedited original filesystem and its explorer.

## Access and laziness

Filesystems and their `Node` handles expose `node`, `listdir`,
`read`, and `locate`. Handles also expose `path` and `kind`. An empty path or `/`
names the namespace root. Repeated separators and `.` normalize; `..` is rejected.
Handle-relative `node(path)` binds a descendant. Selections are separate from paths.

Construction and handle creation do not read data. Untouched subtrees delegate
reads, listings, and metadata inspection directly to their source. Editing a parent
records its child handles once along the edited path. Readers may need data reads
to discover those names: JSON generally loads the file, while HDF5 can inspect
structure and NPZ can list arrays from their headers without reading dataset values.
Expanded groups keep their logical child
names until explicitly edited; there is no automatic rediscovery or refresh.

`read(path, selection=None)` materializes the current logical subtree. Edited
groups traverse their recorded children, so whole-group reads agree with individual
paths and unmounted nodes stay absent. Numeric values keep their reader's native
types and selection behavior: JSON supports list slicing; HDF5 supports direct
multidimensional selections. No implicit NumPy conversion or value cache is added.

`locate(path, selection=None)` returns `list[SourceLocator]` without reads or
structure discovery. Each record holds an absolute `file_path`, an optional
`infile_path`, and `selection`. Namespace aliases preserve those source coordinates,
even when filesystems use the same relative filename under different roots.
An untouched subtree can remain one locator; edited groups collect dependencies
from their current children. Repeated source uses remain repeated. Use `listdir`
to inspect the current namespace.

## Scope

Reference discovery belongs to user code. Deferred resolvers, automatic refresh,
stacking, array composition, and read coalescing are future work. Establish those
semantics with small examples before adding schema or execution abstractions.
