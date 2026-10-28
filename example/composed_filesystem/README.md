# Editable filesystem example

`FileSystem.from_file(path, reader=None)` wraps a reader as a filesystem rooted
at a file's contents. Mount these trees to compose a larger filesystem, then edit
or clone any subtree. Values remain read-only in their source files. File-backed
roots, composed filesystems, and clones use the same implementation.

The filesystem does not interpret strings as links. The example chooses
which stored references to resolve and which nodes to replace.

## Run the examples

With the checkout installed in the active Python environment and Tk available:

```bash
python3 example/composed_filesystem/mount_all.py
python3 example/composed_filesystem/resolve_and_merge.py
python3 example/composed_filesystem/interpret_payloads.py
```

Each script saves its composition next to itself as a specification
(`mount_all.json`, `resolve_and_merge.json`) and reuses it on later
runs instead of composing again. Pass `--rebuild` to compose and save it anew.

`mount_all.py` browses the physical filesystem. `resolve_and_merge.py` clones it,
replaces the partition reference fields with the referenced files, and browses a
reference table with one row per partition; selecting a row opens its group below. Expand groups
on the left or double-click children on the right. Leaf values get short previews
as they appear in the right pane; group contents wait until opened. Double-click
a dataset in the right pane to open its formatted text popup. OS directories show `📁`, file roots `📄`, other groups `⊞`, datasets
`▦`, and scalars `·` in both panes; only groups have expansion arrows. Children are inspected one level at a time. Returning to a parent
shows remembered short text previews; full values are not cached. Relocated mount boundaries also show `↪`, so relocated file roots show `↪ 📄`.

## Fixtures

All paths are relative to `example/composed_filesystem/data/`:

```text
partitions.json
payloads.json
metadata/{alpha,beta,gamma}.json
blocks/{alpha,beta,gamma}_{first,second}.json
```

`partitions.json` contains three entries under `partitions`. Each has `scalar_a`,
`scalar_b`, a length-3 `vector`, a `(2, 3)` `matrix`, a `file_ref` string, and an
ordered `file_refs` list. Metadata files contain `description`, `label`, `weight`,
and `coefficients`. Block files contain `description`, `block_index`, `coordinates`,
and `values`. `payloads.json` holds a `description` and `frames`, a base64-encoded
NPZ archive with `positions` `(3, 4, 2)` and `box` `(3,)`. The distinct fixture
values help verify source order and selection.

## Files as filesystem roots

```python
from lazyfields.filesystem import FileSystem

contents = FileSystem.from_file("example/composed_filesystem/data/metadata/alpha.json")
contents.read("coefficients", 0)  # 1.01

fs = FileSystem()
fs.mount("metadata/alpha.json", contents.node())
fs.read("metadata/alpha.json/coefficients", 0)  # 1.01
```

The reader is inferred from the filename unless supplied explicitly. Neither
`from_file` nor mounting reads file contents. `FileSystem(paths, relative_to=root)`
is shorthand for wrapping and mounting each relative filename this way. Repeated
filenames are registered once; overlapping file paths are rejected before mounting.

Every source is an object interpreted by a `View`: its `load`, and optionally
`read`, `inspect` and `inspect_node`, answer for the object at a key. A `Reader`
is a `View` whose object is a file's path, selected by its `suffixes`.
`FileSystem.from_value(value)` roots a filesystem at an in-memory object, by
default with `MAPPING`, which treats mappings and lists of mappings as groups.
Pass `view=` to interpret other objects; give it `inspect_node` to decide which
nodes are groups. In-memory sources have no files to locate and cannot be
exported with `to_spec()`.

## Interpreting stored values

A value inside a file can hold structure of its own, such as an archive encoded
in a JSON string. Register a view under a name, and set a node's `"view"`
attribute to that name: the node becomes a group whose children are what the
view finds in its value, and traversal, reads and selections continue inside it.
`interpret_payloads.py` decodes `payloads.json/frames` this way:

```python
register_view("npz-base64", View(npz_base64))  # npz_base64(text) -> {name: array}
fs = FileSystem.from_spec(
    {"path": "payloads.json", "children": {"frames": {"attrs": {"view": "npz-base64"}}}},
    relative_to=root,
)
fs.read("frames/positions", (0, slice(2)))
```

The node's value is read and decoded on each request below it. Its locations
continue inside the file, as `payloads.json` at `frames/positions`, so digests
follow the file. Only the attribute is exported; register the same name before
loading the specification. An unregistered name raises `KeyError` when the node is
used. Nodes below an interpreted node cannot be edited.

`FileSystem.from_directory(path, *patterns)` binds a directory and mounts the
files matching glob patterns that registered readers support, at their physical
relative paths. Unsupported files and directories are skipped. Use `"*.json"` to
select JSON files, or `"**/*.json"` to include nested folders. Without patterns
the directory is empty. Construction does not read file contents.

The directory keeps its patterns in `to_spec()`, so rebuilding the specification
discovers the currently matching files; edits are kept as changes beside them.

## Clone, mount, replace, unmount

```python
from pathlib import Path
from lazyfields.filesystem import FileSystem

root = Path("example/composed_filesystem/data").resolve()
fs = FileSystem(
    sorted(path.relative_to(root) for path in root.rglob("*.json")),
    relative_to=root,
)
view = fs.clone()

path = "partitions.json/partitions/alpha/file_ref"
reference = view.read(path)  # 'metadata/alpha.json'
view.replace(path, fs.node(reference))

view.read(f"{path}/coefficients", 0)  # 1.01
fs.read(path)  # 'metadata/alpha.json'
```

`mount` requires a vacant path. Mounting over an existing node raises
`FileExistsError`; use `replace` to replace an existing entry. Missing parent groups are created,
but mounting a child beneath a scalar or dataset raises `NotADirectoryError`.
Unmounting a group removes its whole logical subtree. Physical files are unchanged.
Mounting, replacing, or unmounting the root is rejected; use `clone(path)` to choose a subtree root.

`replace(path, source)` prepares an independent mounted copy and swaps it into an
existing path in one namespace update. Preparation failures leave the old attachment
intact. It raises `KeyError` for a missing destination and does not create parents.
The entry keeps its place in its parent's listing. The source can be inside the
subtree being replaced.

Use `FileSystem()` for an empty namespace. `clone(path)` returns another
`FileSystem` rooted at that subtree. A mount target must be a node obtained
with `node(path)` from any filesystem:

```python
view = fs.clone("partitions.json/partitions/alpha")
view.replace("file_ref", fs.node("metadata/alpha.json"))

combined = FileSystem()
combined.mount("alpha", view.node())
combined.mount("metadata", fs.node("metadata"))
```

`mount`, `replace`, and `unmount` work on the original filesystem too. Cloning or mounting a
node copies its current namespace structure, including edits. Subsequent namespace
changes are independent. Immutable physical source bindings remain shared and values
are read afresh; dataset values are never copied. Rooted scalar and dataset clones
support the same native reads and selections as their sources.

## Reference resolution in the demo

`resolve_and_merge.py` registers only `partitions.json` and clones its `partitions` subtree,
so alpha, beta, and gamma appear directly at the view root. It reads the partition
mapping, resolves references relative to the parent of `partitions.json`, and calls
`replace`. Referenced files are wrapped with `FileSystem.from_file`. It mounts the
single `file_ref` target directly. It replaces `file_refs`
with a merged group for each partition: descriptions get filename prefixes,
block indices are collected into a list, coordinates come from the first block,
and values are stacked along a new first axis. The values shape is `(2, 2, 3)`.

```text
view/
├── alpha/
│   ├── scalar_a
│   ├── scalar_b
│   ├── vector
│   ├── matrix
│   ├── file_ref/                  metadata/alpha.json
│   │   └── coefficients
│   └── file_refs/
│       ├── alpha_first_description
│       ├── alpha_second_description
│       ├── block_index            [0, 1]
│       ├── coordinates
│       └── values                 stacked from both block files
├── beta/                         same structure
└── gamma/                        same structure
```

The tree is abbreviated; mounted files retain all their fields. Raw references
remain accessible through the unedited original filesystem and its browser.

## Access and laziness

`fs.stat(path)` and `node.stat()` expose the physical file's `os.stat_result`.
`fs.digest(path)` and `node.digest()` return a hex string that changes when a file
beneath the node changes, is deleted, or the namespace below it is edited. Nodes
inside a file share that file's size and modification time, so any change to the
file changes all of them. Only file stats are read, and only below the node.
File-backed descendants inherit their source file's stats; groups composed from
multiple sources have no single file stat and return `None`. Stats are read on
request and are not cached. `locate(path)` supplies source paths when aggregate
information is needed, such as the latest source modification time for a table row.

Filesystems and their `Node` handles expose `node`, `listdir`, `info`,
`read`, and `locate`. Handles also expose `path` and `origin`. An empty path or `/`
names the namespace root. Repeated separators and `.` normalize; `..` is rejected.
Handle-relative `node(path)` binds a descendant. Selections are separate from paths.

File registration and handle creation do not read data. The merge factory inspects
immediate source keys once; array computation waits until reading. Untouched subtrees delegate
reads, listings, and metadata inspection directly to their source. Editing a parent
records its child handles once along the edited path. Readers may need data reads
to discover those names: JSON generally loads the file, while HDF5 can inspect
structure and NPZ can list arrays from their headers without reading dataset values.
Expanded groups keep their logical child
names until explicitly edited; there is no automatic rediscovery or refresh.

A merged node's attributes follow its rule, without guessing. A node taken whole from one
source, by preference, by prefix, or as the only occurrence of its key, keeps that node's
attributes. A computed dataset's `info()` has only the `dtype`, `shape`, and `type` its operation
gives, worked out from the inputs' metadata without reading their values: whether an
input's attribute, such as a length, holds for the result, sums, or no longer applies
depends on its meaning. The merged group itself has none. A specification's `attrs` can
add any of them.

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

## Composition specifications

Every filesystem can export its current composition as a plain, JSON-compatible
dictionary. No operation history, parent pointers, Python source, or pickle is
required:

```python
import json

# After constructing view in resolve_and_merge.py:
spec = view.to_spec(relative_to=root)
Path("composed.json").write_text(json.dumps(spec, indent=2))

restored = FileSystem.from_spec(
    json.loads(Path("composed.json").read_text()),
    relative_to=root,
)
restored.read("alpha/file_refs/values")
```

`to_spec()` uses the filesystem's `relative_to` directory as the base by default;
pass `relative_to` to choose another. `from_spec()` resolves paths against the
supplied `relative_to` directory, or the current working directory. The base
itself is not saved. Export performs no source reads, metadata inspection, or
directory discovery, and returns a detached dictionary; call `to_spec()` again to
capture later edits.

Each node is bound in one of three ways, or is a virtual group:

- `path` names a position: a directory, a file, or a place inside a file, such
  as `"partitions.json/partitions"`. It is relative to the parent's position, or
  to the base when it starts with `/`; `..` steps out of a group, a file or a
  directory. A file's reader follows its suffix, or the parent's inside the same
  file, unless `reader` names one.
- `mount` lists glob rules, `[{"glob": "blocks/*.json"}]`, that mount the matching
  supported files of the node's directory at their relative paths.
- `merge` with `sources` merges the children of `sources`: an operation such as
  `"stack"` (with `axis`) gives one dataset, and a mapping of keys to rules, with
  `"*"` for unlisted keys, merges groups key by key.

A virtual group, and a merge, take their parent's position. Any node can carry
`attrs`. `children` then changes the children its binding gives, by name: `null`
removes one, a patch changes or rebinds one in place, and a new name adds one.
Unchanged children are omitted. For example, `FileSystem.from_directory(root,
"**/*.json")` with `metadata/alpha.json/coefficients` unmounted and a file added exports:

```json
{
  "mount": [{"glob": "**/*.json"}],
  "children": {
    "metadata": {"children": {"alpha.json": {"children": {"coefficients": null}}}},
    "extra": {"path": "blocks/alpha_first.json"}
  }
}
```

A patch updates `attrs` name by name, where `null` removes an attribute. Binding
fields in a patch rebind that child in place, keeping its attributes; `null`
clears a binding field the new binding lacks, as when a field is replaced by a merge:

```json
{"file_refs": {"path": null, "merge": {"values": "stack", "*": "prefer"},
               "sources": {"children": {"first": {"path": "/blocks/alpha_first.json"}}}}}
```

Restoring discovers child names wherever a patch edits a source group, and
re-resolves merge keys and mount globs. Readers without metadata support may load
file contents for this discovery; array computations remain lazy. Captures,
conditions, `at` placement, definitions and pattern selectors are not supported
and raise an error.

Facts derived from the source are kept apart from `attrs`, in
`filesystem.info(path)`: its `kind`, `children`, and a dataset's `dtype` and
`shape`. HDF5 and NPZ give those from metadata; other readers load the dataset,
and then also give its Python `type`, such as `matplotlib.figure.Figure`. A file
root's `format` is its reader's, such as `"pdb"`, and a directory's is
`"directory"`, both known without reading. `info(path, full=False)` skips child
names, `dtype` and `shape`, so it reads nothing. Relocation is not saved: `info`
reports it where a subtree's source position first differs from its namespace
position, not below a relocated or merged node.

Viewers in the browsers decide whether they fit from a node's `info`, including
its `attrs` and children's names; its `viewer` attribute names the one opened by
default, and listed first, when it fits.

Custom readers need explicit names on both sides:

```python
readers = {"custom": custom_reader}
spec = filesystem.to_spec(readers=readers)
restored = FileSystem.from_spec(spec, readers=readers)
```

Readers inferred from the current suffix registry need no name; restoration uses
that registry again. An unregistered reader override without a supplied name
raises an error instead of silently changing behavior. No reader callbacks are
serialized.

## Scope

Reference discovery belongs to user code. Per-key merge rules provide collection,
prefixing, source preference, stacking, and concatenation. Deferred resolvers,
automatic refresh, and read coalescing are not implemented.
