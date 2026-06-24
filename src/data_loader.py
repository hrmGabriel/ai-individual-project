from pathlib import Path

import pandas as pd

NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        na_values="?",
        low_memory=False,
    )

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df
