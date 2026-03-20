from .persistence import hdf5_load, json_load, pickle_load
from .store import create_reference_dict, create_reference_table, register_accessors as _register_accessors

__all__ = ["create_reference_dict", "create_reference_table", "pickle_load", "hdf5_load", "json_load"]
__version__ = "0.1.0"

_register_accessors()
