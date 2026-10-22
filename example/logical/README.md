# Logical asset collection: revised design proposal

Status: `AssetsCollection`, `AssetNode`, and `PhysicalLocator` implement the thin
physical filesystem. The logical filesystem, schema, builder, and view definitions
below are proposals, not implemented. Coordinate transformation remains separate
from building logical nodes, with one schema definition driving both.

## Model and boundaries

A physical locator identifies data as `(physical_file, infile_path, selection)`.
A logical locator uses the same coordinates in a namespace defined by the user.
Both layers are read-only filesystems: groups have named children, and datasets
and scalars are leaves. One logical request can require zero, one, or several
physical locators. Dataset selection is an operation on a node, not part of its
filesystem path.

`LogicalAssetCollection` composes `AssetsCollection` and `LogicSchema`. The schema
specifies what logical assets exist and how their contents relate to sources.
A builder binds those definitions to physical filesystem nodes to create a lazy
logical filesystem. A logical asset is a designated group or dataset within that
filesystem; it does not need its own traversal model. Reading follows its
definition; transforming a
locator exposes its physical dependencies without assembling the data.

```text
LogicSchema + AssetsCollection
              |
       LogicalAssetBuilder
              |
    LogicalAssetCollection (logical filesystem)
       |              |
  read(request)   transform(request)
       |              |
 requested data   physical locator list
```

The same view definitions drive both operations. There should not be separately
maintained rules for transforming a stack and constructing that stack.
The builder may also accept explicit source locators when constructing a view,
but it needs a composition definition: the same locator list can describe a stack,
a namespace, or several aliases of the same value.

Keep the first implementation read-only, Python-defined, and unoptimized.
Do not integrate with reference tables or pipes until these semantics work.

## Physical fixtures

All paths below are relative to `data/`.

```text
data/
├── partitions.json
├── metadata/
│   ├── alpha.json
│   ├── beta.json
│   └── gamma.json
└── blocks/
    ├── alpha_first.json
    ├── alpha_second.json
    ├── beta_first.json
    ├── beta_second.json
    ├── gamma_first.json
    └── gamma_second.json
```

`partitions.json` contains `alpha`, `beta`, and `gamma` under `partitions`. Each
entry contains `scalar_a`, `scalar_b`, `vector`, `matrix`, `file_ref`, and
`file_refs`. The vector has length 3 and the matrix has shape `(2, 3)`; their
names describe their structures without implying a relationship between them.

Metadata files contain `description`, `label`, `weight`, and `coefficients`.
Block files contain `description`, `block_index`, `coordinates`, and `values`.
Each block has 2 coordinates and a `(2, 3)` value matrix. The first and second
blocks have indices 0 and 1. All six matrices have distinct values.
These generic fields imply no units.

## Implemented physical filesystem

Physical files are the simplest mounts: combine their registered relative paths
with the infile paths exposed by their readers. Directories are synthesized from
registration; file contents are discovered lazily. Nothing is copied or renamed.

```text
physical/
├── partitions.json/                  file mount, exposed as a group
│   └── partitions/
│       └── alpha/
│           ├── scalar_a              scalar
│           ├── vector                dataset
│           └── matrix                dataset
├── metadata/
│   └── alpha.json/
│       ├── label                     scalar
│       └── coefficients              dataset
└── blocks/
    └── alpha_first.json/
        ├── block_index               scalar
        └── values                    dataset
```

Build this filesystem with the checked-in fixtures:

```python
from pathlib import Path
from lazyfields.assets import AssetsCollection

root = Path("example/logical/data").resolve()
physical = AssetsCollection(
    sorted(path.relative_to(root) for path in root.rglob("*.json")),
    relative_to=root,
)
physical.listdir()  # ('blocks', 'metadata', 'partitions.json')
physical.listdir("partitions.json/partitions")  # ('alpha', 'beta', 'gamma')
node = physical.node("partitions.json/partitions/alpha/vector")
node.read(slice(0, 2))  # [0, 0.5]
node.kind  # 'dataset'
node.locator
# PhysicalLocator(file_path=Path('partitions.json'),
#                 infile_path='partitions/alpha/vector', selection=None)
```

Run `python3 example/logical/use_assets.py` from the repository root for an
executable physical-only example. The checkout must be installed in the active
Python environment, as for the other examples.

| Operation | Physical behavior |
| --- | --- |
| `node(path)` | Bind a lazy handle; infile existence is checked only when accessed |
| `listdir(path)` | Return immediate names; datasets and scalars have no children |
| `read(path, selection=None)` | Read data using the mounted reader |
| `locate(path, selection=None)` | Return one explicit physical locator without I/O |

Handles expose `path`, `locator`, `kind`, `read()`, and `listdir()`. Directories
and file mounts are groups. Infile mappings are groups, atomic values are scalars,
and other values are datasets. `kind` uses reader metadata when available; fallback
readers can load the infile value. Shape and dtype metadata are not yet exposed.

The file boundary comes from registration, never an extension heuristic. A path
such as `blocks/alpha_first.json/values` internally retains the filename and
infile path separately. Overlapping file mounts are rejected. Filesystem paths
may start with `/` to name this namespace's root, never a host path. `..` is
rejected; registered filenames must be relative and remain below `relative_to`.
Repeated separators and `.` are normalized, so they are reserved path syntax.

Reading a physical directory recursively returns a mapping of its registered
children. File/group reads return their mappings. Arrays remain dataset leaves;
selection is passed unchanged to the reader. Iteration and `paths` list registered
filenames without reading data. There is one filesystem access mode, with no
collection-level split-path `get`, `items`, or bracket dispatch. Paths always
include their file mount, regardless of the number of registered files.
For JSON, one-dimensional list slicing works; tuple array indexing still requires
an appropriate reader. No implicit NumPy conversion is added to physical access.

Constructing handles, resolving file boundaries, and listing physical directories
perform no reads. Listing and classification use `Reader.info()`. Its optional
`inspect_node(file, infile_path)` callback returns `NodeInfo(kind, children)`;
HDF5 inspects groups and datasets without reading their values, and NPZ inspects
array headers. Archive readers delegate metadata inspection to the inner reader
after extraction. With no callback, root listing reuses `Reader.items()` when a
fast inspector exists; other requests classify a value obtained through the reader.
JSON generally loads the file; HDF5 can read selected datasets directly.
Values, listings, and classifications are not cached. Physical reference strings
stay scalar leaves; logical link following belongs to the other filesystem.

Implementation boundaries keep the collection small: `filesystem.py` owns the
mount index, node handles, and locator record; `AssetsCollection` binds mounts to
readers; `Reader` owns all format access and metadata
fallbacks. Mount lookup follows path components without scanning registered files
or constructing a `Path` for every prefix. Local reader suffixes are sorted once
during registration. No complete infile tree is built eagerly.

## Define views before transforming requests

A small set of composable definitions covers the examples:

| Definition | Meaning |
| --- | --- |
| `Physical(file, path=None)` | Expose one physical file or subtree |
| `Ref(logical_id)` | Refer to another logical asset |
| `Namespace(children)` | Expose named child views under path prefixes |
| `Stack(sources, fields)` | Stack declared fields from ordered sources on a new first axis |
| `Mount(base, children)` | Add explicitly named child views to an existing subtree |

These are proposed definition records, not a required class hierarchy. A
partition is simply `Physical(file, path=prefix)`; it does not need a special
file class. A key merge is a `Namespace`; it does not need an array merge rule.
Keep `Physical` and `Ref` distinct so aliases cannot be confused with filenames.

`Physical(file, path)` mounts a node from the physical filesystem into the logical
filesystem. `Ref` mounts a node from the logical filesystem. A namespace creates
groups; a stack creates composed dataset leaves per declared field. The same
physical node may appear at several logical paths without copying its data.

For example, all of the following can coexist as different logical assets:

```python
schema = LogicSchema({
    "partitions/alpha": Physical("partitions.json", "partitions/alpha"),
    "metadata/alpha": Physical("metadata/alpha.json"),
    "blocks/alpha": Stack(
        sources=[
            Physical("blocks/alpha_first.json"),
            Physical("blocks/alpha_second.json"),
        ],
        fields=["coordinates", "block_index", "values"],
    ),
    "named_blocks/alpha": Namespace({
        "first": Physical("blocks/alpha_first.json"),
        "second": Physical("blocks/alpha_second.json"),
    }),
})
logical = LogicalAssetCollection(assets, schema)
```

Repeat for beta and gamma, manually first. Schema helper functions can generate
these definitions; discovery from physical data is a later explicit operation.
Logical IDs are opaque names, not physical filenames or paths to parse implicitly.

`Stack` operates per declared field. For a pair of blocks, `coordinates`,
`block_index`, and `values` have logical shapes `(2, 2)`, `(2,)`, and `(2, 2, 3)`.
Descriptions are excluded. Equal-shaped numeric source values are required per
field; missing keys and incompatible shapes are errors when evaluated or explicitly
validated. Concatenation along an existing axis is a separate future operation.

## Proposed logical tree and mounting lifecycle

With the partition mounts and the link mounts described below, one logical asset
could expose this tree. Beta and gamma have the same structure:

```text
logical/
└── partitions/
    └── alpha/                        mounts partitions.json/partitions/alpha
        ├── scalar_a                  physical scalar
        ├── scalar_b                  physical scalar
        ├── vector                    physical dataset (3,)
        ├── matrix                    physical dataset (2, 3)
        ├── file_ref                  raw reference scalar
        ├── file_refs                 raw reference list
        ├── metadata/                 deferred physical-file mount
        │   ├── label                 scalar
        │   └── coefficients          dataset (3,)
        ├── blocks/                   deferred namespace mount
        │   ├── first/
        │   │   └── values            physical dataset (2, 3)
        │   └── second/
        │       └── values            physical dataset (2, 3)
        └── combined/                 stack view of those same block nodes
            └── values                composed dataset (2, 2, 3)
```

The tree is abbreviated: metadata and block groups also contain their other
declared fields. `combined` is an optional additional mount; the standalone
`blocks/alpha` and `named_blocks/alpha` definitions above can coexist as aliases
or alternate organizations. Logical structure need not mirror physical files.

Use the same `node`, `listdir`, and `read` interface for both filesystems.
Read using a complete namespace path and bind handles with `node(path)`.
Keep the asset boundary explicit in the schema; do not infer it from slashes in
the ID. A proposed logical `locate` returns a list of
physical locators, while physical `locate` returns its single locator.

Separate three operations: resolving a mount target, discovering a group's child
names, and reading a leaf value. An explicit mount can bind immediately without
reads. A deferred mount stores a resolver and context; traversal invokes it only
when that mount is reached. Listing the parent can expose the mount's name without
resolving its target. A resolver returns a node definition, including a namespace
or stack, rather than assembling values.

Proposed logical lifecycle methods are `mount(path, definition)`,
`resolve(path, recursive=False)`, and `refresh(path)`. Resolving recursively walks
the structure but does not intentionally materialize dataset leaves; readers may
still need data reads to discover structure. `read(path)` explicitly materializes
the requested subtree. For eager setup, call `resolve` after mounting. For lazy
setup, wait for traversal. Reject collisions unless replacement is explicit.

Initially retain resolved target definitions until explicit refresh. Refresh
invalidates descendant and dependent logical resolutions, including aliases;
this is distinct from caching dataset values. Later mount changes must likewise
invalidate dependent resolutions. A mount encountered again on the active
resolution chain is a cycle; sharing a resolved source is valid.
These mutation and resolution methods belong to the proposed logical filesystem;
the implemented physical collection has fixed mounts supplied at construction.

## Public access and unspecified coordinates

```python
logical.read("partitions/alpha/vector", slice(0, 2))  # [0, 0.5]
logical.read("blocks/alpha/values", (slice(None), 0, slice(0, 2)))
# [[1.0, 1.01], [1.1, 1.11]] -- values shown as lists for readability
logical.read("blocks/alpha/values", (1, 0, slice(0, 2)))  # [1.1, 1.11]
logical.read("named_blocks/alpha/first/values", (0, slice(0, 2)))
# [1.0, 1.01]
```

Read paths follow the physical filesystem convention: an empty path or `/`
names the root; `read()` materializes its namespace hierarchy. Reading
`partitions/alpha` returns that logical asset's complete mapping. A selection
applies to one dataset path; selecting a group is an error. Missing paths are
errors, with no implicit skipping. `selection=None` requests the complete value.
`node(path)` obtains a lazy handle with `read(selection)` and `listdir()`.

For the separate coordinate transformation interface, proposed `None` meanings
remain broader:

| Coordinate | Meaning when `None` |
| --- | --- |
| Logical file | All registered logical assets, in registration order |
| Infile path | The whole logical asset or mapped subtree |
| Selection | The whole value at that path |

`transform(None, path, selection)` resolves dependencies across those assets;
it is not an alternative data-reading mode. Use `paths` or iteration to list
registered IDs without reading data. Data-dependent partition discovery must be
requested explicitly, rather than happening during construction.

Selection follows the logical value's axes. Initially support integers, slices,
and tuples of those, including negative indices, steps, and empty slices. Integer
indices remove axes; slices preserve them and their order. Defer fancy indexing,
boolean masks, new axes, and array-window partition definitions.

Numeric arrays use NumPy semantics in the logical layer. Proposed result policy:
fields declared by `Stack` are NumPy arrays, including scalar fields stacked into
vectors; ordinary unselected JSON values keep their native types. Multidimensional
selection on a numeric JSON list converts it to an array before selection. This
requires an explicit dependency decision before implementation. Physical readers
keep their current behavior; JSON lists cannot currently accept tuple indexing.

## Transform the complete locator

Use one public contract:

```python
transform(logical_file, logical_infile_path, logical_selection)
# -> list[PhysicalLocator]
# PhysicalLocator = (physical_file, physical_infile_path, physical_selection)
```

Each output is a complete triple. A file may depend on the virtual path or the
selected file axis, so file/path/selection are not independent transformations.
The earlier `get_file`, `get_infile_path`, and `get_selection` could be private
helpers inside a rule. They should not be three mandatory public stages: every
stage would need the full context, and different view compositions can require
different resolution orders. Compose view rules instead of enforcing that order.

```python
logical.transform("partitions/alpha", "vector", slice(0, 2))
# [("partitions.json", "partitions/alpha/vector", slice(0, 2))]

logical.transform("blocks/alpha", "values", (slice(None), 0, slice(0, 2)))
# [
#     ("blocks/alpha_first.json", "values", (0, slice(0, 2))),
#     ("blocks/alpha_second.json", "values", (0, slice(0, 2))),
# ]

logical.transform("blocks/alpha", "values", (1, 0, slice(0, 2)))
# [("blocks/alpha_second.json", "values", (0, slice(0, 2)))]
```

The transform describes dependencies. It does not return an assembly callable.
The builder retains the original request and schema definition to preserve axis
placement, source order, repeated sources, and residual selection. Internally,
bindings associate logical uses with physical reads; a deduplicated list alone
cannot preserve this information. These bindings are builder bookkeeping, not
additional assembly behavior in the locator triple.

A physical selection can conservatively cover more data than requested. In that
case the builder must know the residual extraction. The examples above show the
exact selection translation possible for these simple views; a baseline executor
may load the whole physical file and evaluate those selections locally.

## Links are configured mounts

Link resolution is injected. A resolver can use an explicit mapping, a naming
convention, or a reference stored in a physical file. A link may target a physical
subtree, a logical partition, a namespace, or a stack. Strings do not become links
merely because they resemble filenames.

For the fixtures, add logical child mounts `metadata` and `blocks` to each
partition, leaving `file_ref` and `file_refs` as ordinary stored fields:

```python
logical.read("partitions/alpha/file_ref")  # "metadata/alpha.json"
logical.read("partitions/alpha/metadata/coefficients", 0)  # 1.01
logical.read("partitions/alpha/blocks/values", (1, 0, slice(0, 2)))
# [1.1, 1.11]
```

The injected resolver for `metadata` reads `file_ref`; the resolver for
`blocks` reads the ordered `file_refs` list and supplies a stack definition.
References stored in a physical file are relative to that file's parent, not the
logical alias. A resolver returning `Ref(...)` uses an explicit logical ID instead.

Mount names are schema choices. Mounting at a stored key such as `file_ref`
is also possible, but must explicitly replace that logical key: both exact access
and traversal then see the mounted target, and physical access supplies the raw
reference. Avoid a key that returns a string on exact access but behaves as a
mapping during traversal. Whole-asset reads must agree with individual path reads.

Resolving dynamic links can require metadata reads before physical dependencies
are known. Therefore `transform()` need not be free of I/O. Constructing explicit
schema definitions and binding handles should perform no reads. Share metadata
reads within an operation, and detect cycles using the active resolution chain;
shared targets are valid and must not be mistaken for cycles.

## Fewest requests is a separate execution policy

The objective is the minimum number of physical requests, not the minimum bytes
read. Keep exact logical dependencies separate from the continuous regions chosen
to fetch them. A stepped selection can be covered by a continuous range and then
selected locally. Logical paths have no inherent numeric continuity; shared-parent
subtree reads need a separate policy. Array rectangles also need a precise
allowed-region definition before optimizing them.

For a first, deliberately unoptimized executor, allow whole-file reads. Load each
required physical file at most once per `read()` operation, share it across all
logical uses in that operation, and perform path/selection operations in memory.
One whole-file read per required file minimizes data-read calls under that policy,
at the cost of reading unused content. Count discovery and dynamic-link reads too;
reuse a loaded file when it is later needed for data. This does not require a
persistent cache or changes to `AssetsCollection`'s no-cache contract.

The transform's exact locator list is not necessarily the executor's read list.
A later policy can use direct reader selections, merge adjacent requests, or
cover several selections with one larger region. Retain a mapping from each
logical use to the chosen physical read and local extraction. The optimized and
whole-file executors must return the same result.

Reader calls are the initial unit of request counting. Actual disk reads or
remote requests depend on the reader and cannot be inferred from locator count.
The current JSON reader loads a file even for path access; HDF5 supports direct
selection. Shape metadata is not yet exposed by `Reader`, so avoid promising
I/O-free shape resolution or skipping unrelated sources for every indexing case.
For an explicit nonnegative source index in a known stack, dependency resolution
can identify the single source without loading other values. Full cross-source
shape validation is a separate operation or happens when those sources are read.

## Minimal components

| Component | Responsibility |
| --- | --- |
| `AssetsCollection` | Thin binding between the mount index and selected readers |
| `AssetNode` | Implemented lazy physical node handle; common interface for future logical nodes |
| `Reader` / `NodeInfo` | Physical reads and optional metadata-only node inspection |
| `LogicSchema` | Logical IDs, composable view definitions, injected resolvers |
| `LogicalAssetBuilder` | Build logical nodes from physical/logical mounts; retain composition and read bindings |
| `LogicalAssetCollection` | Proposed logical filesystem; traversal, mounts, resolution, `get`, and `transform` |
| `PhysicalLocator` | Implemented physical file/path/selection triple with no assembly behavior |

Start with definition records and a small interpreter. A separate class for every
partition, merge, or link combination is unnecessary. A physical `Asset` handle,
serialized schema, generic optimizer, and mathematical transformation library are
not prerequisites.

Validate duplicate IDs, child-name collisions, and unsupported definitions when
constructing the schema. Report unknown assets, missing paths, invalid selections,
dangling links, cycles, and incompatible evaluated shapes explicitly, with the
logical request and physical source where available.

## Implementation sequence

1. Physical filesystem foundation is implemented. Next add explicit `Physical`
   mounts, schema registration, logical node traversal, and complete-locator
   transformation. Verify the alpha coordinate example and whole-subtree access.
2. Add the builder and reference executor with per-operation whole-file reuse.
   Verify full reads agree with field reads and repeated uses do not repeat loads.
3. Add `Namespace`, `Ref`, and per-field `Stack`. Test distinct fixture values,
   scalar stacking, source-axis removal, reversed/stepped/empty slices, bounds,
   missing fields, and shape errors. Compare read results with direct evaluation
   of the same fully materialized logical view.
4. Add injected mounts and explicit discovery. Test raw reference access, logical
   target traversal, relative physical paths, shared targets, links to partitions,
   dynamic linked stacks, cycle errors, and consistent whole-asset reads.
5. Add bound handles and a runnable example. Keep the three scenarios visible:
   one physical file to many views, many sources to one view, and linked views.
6. Only then add direct selection execution and request coalescing. Compare their
   results with the reference executor and count reader calls under each policy.

## Decisions still open

- Confirm replacing the three mandatory hooks with one transform over a complete
  locator. Helpers remain possible inside individual view definitions.
- Confirm the proposed `None` rules for broad coordinate transformation.
- Confirm NumPy use and the logical array result policy.
- Confirm separate mount names `metadata` and `blocks` as the fixture convention.
- Confirm whole-file reads as the initial request-minimizing execution policy.

## Mathematical perspective and prior work

A locator dependency mapping can be modeled as a relation between logical and
physical coordinates. Relations compose across intermediate views. Sets of
physical dependencies do not capture ordering or repeated logical placement;
those remain in the view definition and builder. Index offsets and strides can
later use integer-region mappings, without requiring that machinery for paths
and dynamic links.

- [HDF5 Virtual Datasets](https://docs.h5py.org/en/stable/vds.html) map physical
  datasets into a sliceable virtual dataset, including stacked layouts.
- [HDF5 groups and links](https://docs.h5py.org/en/stable/high/group.html) provide
  namespaces and links, including external file targets.
- [Kerchunk aggregation](https://fsspec.github.io/kerchunk/tutorial.html) and
  [fsspec ReferenceFileSystem](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.reference.ReferenceFileSystem)
  provide reference-backed virtual data over existing files, primarily at the
  byte/chunk level.
- [xarray.open_mfdataset](https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html)
  is a precedent for explicit multi-file combination rules.
- [Integer Set Library](https://libisl.sourceforge.io/user.html) represents integer
  sets and relations and is relevant to future array-index transformations.

These are precedents rather than asserted drop-in solutions. The proposed scope
here is a reader-independent namespace over mapping-like files with composable
views and user-defined links.
