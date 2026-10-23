from pathlib import Path

import lazyfields as lf
from lazyfields.extra_readers import CSV_WITH_HEADER
from lazyfields.filesystem import FileSystem


root = Path(__file__).parent
reference_path = root / "reference_table.pkl"
lf.register_reader(CSV_WITH_HEADER(skiprows=2))
data = root / "data"
filesystem = FileSystem(sorted(path.name for path in data.glob("*.csv")), relative_to=data)
lf.create_reference_table(filesystem).to_pickle(reference_path)
