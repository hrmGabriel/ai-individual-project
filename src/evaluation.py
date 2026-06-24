import time
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

T = TypeVar("T")

METRIC_COLUMNS = ["mae", "rmse", "r2", "train_time", "total_time"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def measure_execution(fn: Callable[[], T]) -> tuple[T, float, float]:
    total_start = time.perf_counter()
    train_start = time.perf_counter()
    result = fn()
    train_time = time.perf_counter() - train_start
    total_time = time.perf_counter() - total_start
    return result, train_time, total_time


def aggregate_results(runs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model, group in runs_df.groupby("model"):
        for metric in METRIC_COLUMNS:
            values = group[metric]
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "median": values.median(),
                }
            )

    return pd.DataFrame(rows)


def print_experiment_metadata(metadata: dict[str, Any]) -> None:
    print("=== Metadados do Dataset ===")
    print(f"Linhas originais: {metadata['rows_before']:,}")
    print(f"Linhas removidas: {metadata['rows_removed']:,}")
    print(f"Amostras finais: {metadata['rows_after']:,}")
    print(f"Features ({len(metadata['features'])}): {metadata['features']}")
    print()
    print("=== Parâmetros ===")
    print(f"test_size: {metadata['test_size']}")
    print("Regressão Linear: LinearRegression + StandardScaler")
    print(
        "Random Forest: RandomForestRegressor("
        f"n_estimators={metadata['n_estimators']})"
    )
    print()


def print_summary(summary_df: pd.DataFrame) -> None:
    print("=== Resultados Agregados (30 execuções) ===")
    print(summary_df.to_string(index=False))
    print()
    print("=== Observação ===")
    print(
        "Regressão Linear: variabilidade entre execuções decorre do split "
        "(modelo determinístico)."
    )
    print(
        "Random Forest: variabilidade decorre do split + aleatoriedade do ensemble."
    )
