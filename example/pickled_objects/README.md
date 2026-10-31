# Pickled objects

A pickle can hold objects of any class, such as a pandas DataFrame or a class
only this example defines. lazyfields reads pickles as data: it never imports
those classes or runs their code, unless its pickle reader explicitly trusts
that class (see below). An untrusted object remains an inert record.

From the repository root, with Tk available:

```bash
python3 example/pickled_objects/generate_dataset.py
python3 example/pickled_objects/browse.py
```

`generate_dataset.py` writes `data/results.pkl`: a DataFrame `table`, a `run`
of the script's own `Run` class, a `trajectory` array, a `label`, `plot`, a
matplotlib `Figure`, and `plot_callable`, a plain function.
`run`, `plot`, and `plot_callable` are inert records by default: nothing
imports this example's own code or matplotlib to read them. `run` and `plot`
are still directly usable, though: both are ordinary objects whose pickled
state is just their own `__dict__`, so `run["temperature"]`, `run["started"]`
and `run["positions"]` (a real NumPy array) read out directly, the same shape
a trusted `Run` would have - and so does `plot["lines"]`, `plot["patches"]`,
and the rest of a `Figure`'s real attributes, without ever importing or
running `Run`'s or `Figure`'s own code. `plot_callable` doesn't get this: a
bare function reference pickles as only a name, no state at all, so it stays
the plain, empty type placeholder it always was. `table` is rebuilt as a real
DataFrame even by default, from its already-safe column and index arrays
through pandas' own public constructor - not by trusting anything
pandas-specific; a warning notes what that approximation can't preserve
(extension dtypes, categoricals, a `MultiIndex`, `attrs`, `flags`).
`trajectory` and `label` are plain data (a NumPy array and a string), so they
read, slice, and browse exactly as any other in-memory array or string would.

## A trusted type is just a value

Trust belongs only to `pickle_reader`. Trusting `"matplotlib"` makes every
class under that package unpickle for real, not just `Figure`: a complex
object nests many classes in its pickled state. `trust` also takes a class,
or a `(module, name)` pair, to trust one global exactly.

Trusting a prefix runs that library's own unpickling code on the file, the
same as `pickle.load` would for that code. Configure it only for pickle files
you trust.

`browse.py` mounts the same file three times, side by side, once through each reader:

```python
trusted = pickle_reader(trust=["pandas", "matplotlib"])
naive = pickle_reader(trust=True)  # Ordinary pickle.load semantics: every class runs, for real.
FileSystem.from_spec({"children": {
    "untrusted": {"path": "results.pkl"},
    "trusted": {"path": "results.pkl", "reader": "trusted-pickle"},
    "naive": {"path": "results.pkl", "reader": "naive-pickle"},
}}, relative_to=DATA_DIR, readers={"trusted-pickle": trusted, "naive-pickle": naive})
```

Under `trusted`, pandas and matplotlib are trusted, so `table` is the exact
DataFrame pandas itself would unpickle, rather than `untrusted`'s rebuilt
approximation, and `plot` is a real Figure, handled by the existing Figure
method - `untrusted` has no equivalent for `plot`, since nothing rebuilds a
matplotlib Figure safely yet. `run` stays an inert record under both mounts:
trusting pandas and matplotlib doesn't trust this example's own `Run` class.

## Naive: what plain `pickle.load` would have done

`naive` uses `trust=True`: no allowlist, no records, the same as calling
`pickle.load` directly. Opening anything under it fails - `table`, `plot`,
even `label`, not just `run` - with `Run` not found in whatever script is
running `browse.py`. That's not a bug in the example: a pickle is one object
graph, unpickled in a single pass, so one class `pickle.load` can't import
poisons the entire file, even fields that have nothing to do with it. Listing
`naive`'s contents still works (nothing is loaded just to list it); opening
any one of them is where it fails.

`untrusted` and `trusted` never have this failure mode: an unrecoverable
class just becomes an inert record for that one field - still readable, as
`run["temperature"]` shows above - while everything else keeps reading
normally. Naive loading is strictly worse in two ways at once: it runs any
class's code for real, and a single missing or untrusted one can take the
whole file down with it.
