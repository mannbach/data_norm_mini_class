import pandas as pd

from .constants import (
    FILE_DATA_RAW
)

def read_raw_aarc_data(
        file_path: str = FILE_DATA_RAW
):
    return pd.read_csv(file_path)\
        .convert_dtypes()
