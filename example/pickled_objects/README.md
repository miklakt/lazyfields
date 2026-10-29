# Pickled objects

A pickle can hold objects of any class, such as a pandas DataFrame or a class
only the writing script defines. lazyfields reads pickles as data: it never
imports those classes or runs their code, unless that class is trusted (see
below). An untrusted object is a leaf, showing only its type - `pandas.DataFrame`,
`__main__.Run` - with nothing underneath it to open.

From the repository root, with Tk available:

```bash
python3 example/pickled_objects/generate_dataset.py
python3 example/pickled_objects/browse.py
```

`generate_dataset.py` writes `data/results.pkl`: a DataFrame `table`, a `run`
of the script's own `Run` class, a `trajectory` array, a `label`, and `plot`,
a matplotlib `Figure`. `table`, `run`, and `plot` are all leaves by default:
nothing imports pandas, the script's own classes, or matplotlib to read them.
`trajectory` and `label` are plain data (a NumPy array and a string), so they
read, slice, and browse exactly as any other in-memory array or string would.

## A trusted type is just a value

`plot` shows up live rather than as a leaf: the browser's built-in "Figure"
viewer, in `browser/viewers.py`, registers itself with `trust="matplotlib"`,
so `results.pkl/plot` is a real `Figure` and that viewer renders it as the
plot it is.

There is no standalone function for trusting a type; `register_view` and
`register_viewer` are the only way to grant it, each taking the same `trust=`
argument, so a class only ever unpickles for real alongside something that
can actually use the live object. Trusting `"matplotlib"` makes every class
under that package unpickle for real, not just `Figure`: a complex object
nests many classes in its pickled state, and any that stays untrusted makes
the whole object a leaf too. `trust` also takes a class, or a module and a
class name, to trust one global exactly.

Trusting a prefix runs that library's own unpickling code on the file, the
same as `pickle.load` would: a viewer or view should only ask for it when it
trusts the pickles it will be given.

`browse.py` does the same for `table`, mounting the file a second time as
`readable` with `table` read through a view:

```python
def columns(table):
    """A DataFrame's index and columns, as plain values; ordinary pandas code once pandas is trusted."""
    return dict(table.items()) | {"index": list(table.index)}

register_view("dataframe-columns", View(columns), trust="pandas")
FileSystem.from_spec({"mount": [{"glob": "*.pkl"}], "children": {
    "readable": {"path": "results.pkl", "children": {"table": {"attrs": {"view": "dataframe-columns"}}}},
}}, relative_to=DATA_DIR)
```

```text
readable/table/index          ['s1', 's2', 's3']
readable/table/temperature    [300.0, 325.0, 350.0]
readable/table/energy         [-1.25, -1.1, -0.95]
readable/table/steps          [1000, 2000, 4000]
```

Once pandas is trusted, `table` is a real `DataFrame` and `columns` is
ordinary pandas code - no hand-parsing of pandas' internal pickle layout, and
nothing to maintain across pandas versions.
