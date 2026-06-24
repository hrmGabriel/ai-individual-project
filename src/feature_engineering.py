import pandas as pd

EXCLUDED_COLUMNS = ["Date", "Time", "Datetime", "Global_active_power"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["Datetime"].dt.hour
    df["day"] = df["Datetime"].dt.day
    df["month"] = df["Datetime"].dt.month
    df["weekday"] = df["Datetime"].dt.weekday
    return df


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_columns = [column for column in df.columns if column not in EXCLUDED_COLUMNS]
    X = df[feature_columns]
    y = df["Global_active_power"]
    return X, y, feature_columns
