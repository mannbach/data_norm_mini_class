import pandas as pd

from .constants import (
    FILE_DATA_RAW
)

def read_raw_aarc_data(
        file_path: str = FILE_DATA_RAW
):
    return pd.read_csv(file_path)\
        .convert_dtypes()

def write_raw_aarc_data(
        df: pd.DataFrame,
        file_path: str = FILE_DATA_RAW
):
    df.to_csv(file_path, index=False)
