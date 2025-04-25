import numpy as np

from data_norm_class.constants import (
    FILE_DATA_RAW_FULL,
    FILE_DATA_RAW
)
from data_norm_class.io import (
    read_raw_aarc_data,
    write_raw_aarc_data
)

def main():
    np.random.seed(0)

    print(f"Loading full AARC data from {FILE_DATA_RAW_FULL}...")
    aarc_raw = read_raw_aarc_data(FILE_DATA_RAW_FULL)

    id_deps = np.random.choice(
        aarc_raw["DepartmentId"].unique(),
        size=40,
        replace=False)
    print(f"Sampled departments (N={len(id_deps)}):\n", id_deps)


    aarc_raw_sample = aarc_raw[aarc_raw["DepartmentId"].isin(id_deps)].copy()
    print(f"Sampled AARC data shape: {aarc_raw_sample.shape} (original: {aarc_raw.shape})")
    print(f"Faculty in sampled data: {aarc_raw_sample['PersonId'].nunique()}")

    print("Hiding `PersonName` column...")
    for col in ["PersonName"]:
        aarc_raw_sample[col] = "<hidden>"

    print(f"Writing sampled AARC data to {FILE_DATA_RAW}...")
    write_raw_aarc_data(aarc_raw_sample, FILE_DATA_RAW)

if __name__ == "__main__":
    main()
