import json
from pathlib import Path

import pandas as pd

from . import Reader, value_kind


def CSV_WITH_HEADER(comment: str = "#", key_value_separator: str = ":", skiprows: int = 0, column_separator: str = ",", suffixes: tuple[str, ...] = (".csv",)) -> Reader:
    def metadata(path):
        values = {}
        with Path(path).open() as file:
            for _ in range(skiprows): next(file, None)
            for line in file:
                if not line.startswith(comment): break
                key, value = line[len(comment):].split(key_value_separator, 1)
                value = json.loads(value)
                if value_kind(value) != "scalar": raise ValueError("CSV header values must be scalar.")
                values[key.strip()] = value
        return values

    def load(path):
        table = pd.read_csv(path, comment=comment, skiprows=skiprows, sep=column_separator)
        return {**metadata(path), **{key: table[key].to_numpy() for key in table}}

    def read(path, key, selection=None):
        values = metadata(path)
        value = values[key] if key in values else pd.read_csv(path, comment=comment, skiprows=skiprows, sep=column_separator, usecols=[key])[key].to_numpy()
        return value if selection is None else value[selection]

    def inspect(path, key=None):
        if key is not None:
            if not any(name == key for name, _, _ in inspect(path)):
                raise KeyError(key)
            raise NotADirectoryError(key)
        yield from ((key, "scalar", value) for key, value in metadata(path).items())
        columns = pd.read_csv(path, comment=comment, skiprows=skiprows, sep=column_separator, nrows=0).columns
        yield from ((key, "dataset", None) for key in columns)

    return Reader(suffixes, load, read, inspect, attrs={"file_format": "csv"})
