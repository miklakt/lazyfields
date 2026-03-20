#%%
import json
import pickle
from pathlib import Path

import lazyfields as lf


root_dir = Path(__file__).resolve().parent
results_dir = root_dir / "results"
data_dir = root_dir / "data"
results_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

sample_rows = [
    {
        "some_int_key": 1,
        "some_float_key1": 1.25,
        "some_float_key2": 10.5,
        "some_string_key": "run-a",
        "some_array_key": [0.1, 0.2, 0.3, 0.4],
        "some_other_array_key": [1, 2, 3, 4],
    },
    {
        "some_int_key": 2,
        "some_float_key1": 2.5,
        "some_float_key2": 20.0,
        "some_string_key": "run-b",
        "some_array_key": [1.1, 1.2, 1.3, 1.4],
        "some_other_array_key": [10, 20, 30, 40],
    },
    {
        "some_int_key": 3,
        "some_float_key1": 3.75,
        "some_float_key2": 30.25,
        "some_string_key": "run-c",
        "some_array_key": [2.1, 2.2, 2.3, 2.4],
        "some_other_array_key": [100, 200, 300, 400],
    },
]

for idx, row_data in enumerate(sample_rows, start=1):
    json_path = results_dir / f"result_{idx:03d}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(row_data, fh, indent=2)


for json_path in sorted(results_dir.glob("*.json")):
    with open(json_path, "r", encoding="utf-8") as fh:
        row_data = json.load(fh)

    pickle_path = data_dir / f"{json_path.stem}.pkl"
    with open(pickle_path, "wb") as fh:
        pickle.dump(row_data, fh, protocol=4)


reference_df = lf.create_reference_table(data_dir)

display(reference_df)
#%%
reference_df.iloc[0].pickle_file
#%%
