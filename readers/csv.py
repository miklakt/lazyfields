import csv
import json
from pathlib import Path

import pandas as pd

from . import NodeInfo, Reader, value_kind


def CSV_WITH_HEADER(comment: str = "#", key_value_separator: str = ":", skiprows: int = 0, column_separator: str = ",", suffixes: tuple[str, ...] = (".csv",)) -> Reader:
    def header(path):
        """The comment block's values and the column names, from the top of the file."""
        values = {}
        with Path(path).open() as file:
            for _ in range(skiprows): next(file, None)
            for line in file:
                if not line.strip(): continue
                if not line.startswith(comment):
                    return values, tuple(next(csv.reader([line], delimiter=column_separator)))
                key, value = line[len(comment):].split(key_value_separator, 1)
                value = json.loads(value)
                if value_kind(value) != "scalar": raise ValueError("CSV header values must be scalar.")
                values[key.strip()] = value
        return values, ()

    def load(path):
        table = pd.read_csv(path, comment=comment, skiprows=skiprows, sep=column_separator)
        return {**header(path)[0], **{key: table[key].to_numpy() for key in table}}

    def read(path, key, selection=None):
        values = header(path)[0]
        value = values[key] if key in values else pd.read_csv(path, comment=comment, skiprows=skiprows, sep=column_separator, usecols=[key])[key].to_numpy()
        return value if selection is None else value[selection]

    def describe(path):
        values, columns = header(path)
        yield None, NodeInfo("group", (*values, *columns)), None
        yield from ((key, NodeInfo("scalar"), value) for key, value in values.items())
        yield from ((key, NodeInfo("dataset"), None) for key in columns)

    return Reader(load, read, describe=describe, suffixes=suffixes, format="csv")
