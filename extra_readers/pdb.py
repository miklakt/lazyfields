from pathlib import Path

from ..readers import Reader


# Zero-based slices for the fixed-width PDB ATOM and HETATM records.
_PDB_COLUMNS = {
    "record": (slice(0, 6), str), "atom_serial": (slice(6, 11), int),
    "atom_name": (slice(12, 16), str), "alternate_location": (slice(16, 17), str),
    "residue_name": (slice(17, 20), str), "chain_id": (slice(21, 22), str),
    "residue_number": (slice(22, 26), int), "insertion_code": (slice(26, 27), str),
    "x": (slice(30, 38), float), "y": (slice(38, 46), float), "z": (slice(46, 54), float),
    "occupancy": (slice(54, 60), float), "temperature_factor": (slice(60, 66), float),
    "element": (slice(76, 78), str), "charge": (slice(78, 80), str),
}


def _pdb_metadata(path):
    values, titles = {}, []
    with Path(path).open() as file:
        for line in file:
            record = line[:6].strip()
            if record == "HEADER":
                values.update(classification=line[10:50].strip(), deposition_date=line[50:59].strip(), id_code=line[62:66].strip())
            elif record == "TITLE": titles.append(line[10:70].strip())
            elif record in {"ATOM", "HETATM"}: break
    if titles: values["title"] = " ".join(titles)
    return values


def _pdb_value(line, column):
    part, convert = column
    value = line[part].strip()
    return convert(value) if value else None


def _pdb_load(path):
    columns = {key: [] for key in _PDB_COLUMNS}
    with Path(path).open() as file:
        for line in file:
            if line[:6].strip() in {"ATOM", "HETATM"}:
                for key, column in _PDB_COLUMNS.items(): columns[key].append(_pdb_value(line, column))
    return {**_pdb_metadata(path), **columns}


def _pdb_read(path, key, selection=None):
    if key in _PDB_COLUMNS:
        with Path(path).open() as file:
            value = [_pdb_value(line, _PDB_COLUMNS[key]) for line in file if line[:6].strip() in {"ATOM", "HETATM"}]
    else: value = _pdb_metadata(path)[key]
    return value if selection is None else value[selection]


def _pdb_inspect(path):
    yield from ((key, "scalar", value) for key, value in _pdb_metadata(path).items())
    yield from ((key, "dataset", None) for key in _PDB_COLUMNS)


PROTEIN_DATA_BANK = Reader((".pdb", ".ent"), _pdb_load, _pdb_read, _pdb_inspect)
