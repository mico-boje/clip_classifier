import os
from pathlib import Path


def get_root_path():
    """Get the root path of the project"""
    root_path = Path(os.path.abspath(__file__)).parent.parent.parent
    return root_path

def get_data_path():
    """Get the data path of the project"""
    data_path = os.path.join(get_root_path(), "data")
    return data_path