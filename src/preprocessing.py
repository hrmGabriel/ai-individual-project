import pandas as pd


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows_before = len(df)

    df = df.dropna()
    rows_after = len(df)

    df = df.copy()
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
    )
    df = df.sort_values("Datetime").reset_index(drop=True)

    metadata = {
        "rows_before": rows_before,
        "rows_removed": rows_before - rows_after,
        "rows_after": rows_after,
    }

    return df, metadata
