import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
import pickle
import warnings

import pandas as pd


def is_scalar(value: Any) -> bool:
    return bool(pd.api.types.is_scalar(value))


def _pickle_load(path: str | Path) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _relpath(path: str | Path, base_dir: str | Path | None = None) -> str:
    path = Path(path)
    if base_dir is None:
        return str(path)
    try:
        return os.path.relpath(path, start=Path(base_dir))
    except ValueError:
        return str(path)


def create_reference_dict(
    directory: str | Path = "data",
    columns: list[str] | None = None,
    *,
    reference_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    directory = Path(directory)
    rows: list[dict[str, Any]] = []
    for pickle_path in sorted(directory.glob("*.pkl")):
        try:
            row_data = _pickle_load(pickle_path)
        except Exception as exc:
            print(f"Failed to load {pickle_path.name}: {exc}")
            continue

        if not isinstance(row_data, Mapping):
            print(f"Skipping {pickle_path.name}: expected a mapping row dictionary.")
            continue

        rows.append(_reference_row(row_data, pickle_path, columns=columns, reference_path=reference_path))

    return rows


def create_reference_table(
    directory: str | Path = "data",
    columns: list[str] | None = None,
    *,
    reference_path: str | Path | None = None,
) -> pd.DataFrame:
    reference_df = pd.DataFrame(create_reference_dict(directory, columns=columns, reference_path=reference_path))
    if reference_path is not None:
        reference_df.attrs["reference_path"] = str(reference_path)
    return reference_df


def _reference_row(
    row_data: Mapping[str, Any],
    pickle_path: Path,
    *,
    columns: list[str] | None = None,
    reference_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one reference-table row from one stored row.

    Scalar fields are copied into the reference row, while non-scalar fields
    remain inside the pickled row file.
    """
    row = {key: value for key, value in row_data.items() if is_scalar(value)}
    if columns is not None:
        row = {key: value for key, value in row.items() if key in columns}

    reference_dir = Path(reference_path).parent if reference_path is not None else None
    row["pickle_file"] = _relpath(pickle_path, reference_dir)
    row["non_scalar_keys"] = [key for key, value in row_data.items() if not is_scalar(value)]
    row["creation_time"] = datetime.fromtimestamp(pickle_path.stat().st_ctime)
    return row


class LazyPickleList:
    def __init__(self, files: Iterable[str | Path]):
        self.files = [str(path) for path in files]

    def __getitem__(self, idx: int | slice) -> Any:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self.files)))]
        return _pickle_load(self.files[idx])

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self):
        yield from map(_pickle_load, self.files)


class PickleStorageAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj
        self._base_dir = (
            Path(pandas_obj.attrs["reference_path"]).parent
            if "reference_path" in pandas_obj.attrs
            else None
        )

    def _path(self, pickle_path: str | Path) -> Path:
        path = Path(pickle_path)
        return path if path.is_absolute() or self._base_dir is None else self._base_dir / path

    def __getitem__(self, keys: str | list[str] | slice) -> Any:
        files = self._obj["pickle_file"].map(self._path)
        if keys == slice(None):
            return LazyPickleList(files.tolist())

        if isinstance(keys, str):
            if keys == "pickle_file":
                return files
            return pd.Series((_pickle_load(path)[keys] for path in files), index=self._obj.index)

        if isinstance(keys, list):
            return pd.DataFrame({key: self[key] for key in keys}, index=self._obj.index)

        raise TypeError("Keys must be str, list[str], or [:]")


class PickleStorageAccessorSeries(PickleStorageAccessor):
    def __getitem__(self, keys: str | list[str] | slice) -> Any:
        pickle_file = self._path(self._obj.pickle_file)
        if keys == slice(None):
            return _pickle_load(pickle_file)

        if isinstance(keys, str):
            if keys == "pickle_file":
                return pickle_file
            return _pickle_load(pickle_file)[keys]

        if isinstance(keys, list):
            return {key: self[key] for key in keys}

        raise TypeError("Keys must be str, list[str], or [:]")


def register_accessors() -> None:
    if "pkl" in pd.DataFrame._accessors:
        warnings.warn("DataFrame accessor 'pkl' was already registered; skipping.", stacklevel=2)
    else:
        pd.api.extensions.register_dataframe_accessor("pkl")(PickleStorageAccessor)

    if "pkl" in pd.Series._accessors:
        warnings.warn("Series accessor 'pkl' was already registered; skipping.", stacklevel=2)
    else:
        pd.api.extensions.register_series_accessor("pkl")(PickleStorageAccessorSeries)
