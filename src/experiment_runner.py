import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_raw_data
from src.evaluation import (
    aggregate_results,
    compute_metrics,
    print_experiment_metadata,
    print_summary,
)
from src.feature_engineering import build_xy, engineer_features
from src.linear_regression_model import ModelResult, train_linear_regression
from src.preprocessing import preprocess
from src.random_forest_model import N_ESTIMATORS, train_random_forest
from src.visualization import generate_exploratory_plots, generate_result_plots

SEEDS = list(range(1, 31))
TEST_SIZE = 0.20
PLOT_SEED = 1

MODEL_LINEAR = "linear_regression"
MODEL_RF = "random_forest"


def _evaluate_run(
    y_test: np.ndarray,
    train_fn: Callable[[], ModelResult],
) -> tuple[dict[str, float], ModelResult]:
    total_start = time.perf_counter()
    train_start = time.perf_counter()
    result = train_fn()
    train_time = time.perf_counter() - train_start
    metrics = compute_metrics(y_test, result.y_pred)
    total_time = time.perf_counter() - total_start

    run_metrics = {
        **metrics,
        "train_time": train_time,
        "total_time": total_time,
    }
    return run_metrics, result


def _save_results(
    runs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    metadata: dict,
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    runs_df.to_csv(results_dir / "all_runs.csv", index=False)
    runs_df[runs_df["model"] == MODEL_LINEAR].to_csv(
        results_dir / "linear_regression_runs.csv",
        index=False,
    )
    runs_df[runs_df["model"] == MODEL_RF].to_csv(
        results_dir / "random_forest_runs.csv",
        index=False,
    )
    summary_df.to_csv(results_dir / "summary.csv", index=False)

    with open(results_dir / "experiment_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def run_experiment(
    data_path: Path,
    plots_dir: Path,
    results_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data(data_path)
    processed_df, preprocessing_meta = preprocess(raw_df)
    featured_df = engineer_features(processed_df)
    X, y, feature_columns = build_xy(featured_df)

    print("Gerando gráficos exploratórios...", flush=True)
    generate_exploratory_plots(featured_df, feature_columns, plots_dir)
    print(flush=True)

    run_rows: list[dict] = []
    plot_predictions: dict[str, np.ndarray] | None = None

    print(f"Iniciando experimento com {len(SEEDS)} execuções...", flush=True)
    print(flush=True)

    for seed in SEEDS:
        print(f"Execução seed={seed}...", flush=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=seed,
        )

        X_train_values = X_train.to_numpy()
        X_test_values = X_test.to_numpy()
        y_train_values = y_train.to_numpy()
        y_test_values = y_test.to_numpy()

        lr_metrics, lr_result = _evaluate_run(
            y_test_values,
            lambda: train_linear_regression(
                X_train_values,
                y_train_values,
                X_test_values,
            ),
        )
        run_rows.append({"seed": seed, "model": MODEL_LINEAR, **lr_metrics})

        rf_metrics, rf_result = _evaluate_run(
            y_test_values,
            lambda: train_random_forest(
                X_train_values,
                y_train_values,
                X_test_values,
                seed,
            ),
        )
        run_rows.append({"seed": seed, "model": MODEL_RF, **rf_metrics})

        if seed == PLOT_SEED:
            plot_predictions = {
                "y_test": y_test_values,
                "y_pred_linear_regression": lr_result.y_pred,
                "y_pred_random_forest": rf_result.y_pred,
            }

    runs_df = pd.DataFrame(run_rows)
    summary_df = aggregate_results(runs_df)

    metadata = {
        **preprocessing_meta,
        "features": feature_columns,
        "test_size": TEST_SIZE,
        "n_estimators": N_ESTIMATORS,
        "seeds": SEEDS,
        "plot_seed": PLOT_SEED,
        "models": [MODEL_LINEAR, MODEL_RF],
        "note_linear_regression_variability": (
            "Variabilidade entre execuções decorre do split "
            "(modelo determinístico)."
        ),
        "note_random_forest_variability": (
            "Variabilidade decorre do split + aleatoriedade do ensemble."
        ),
    }

    _save_results(runs_df, summary_df, metadata, results_dir)

    if plot_predictions is not None:
        np.savez(
            results_dir / f"plot_predictions_seed_{PLOT_SEED}.npz",
            **plot_predictions,
        )
        print("Gerando gráficos de resultados...", flush=True)
        generate_result_plots(runs_df, plot_predictions, PLOT_SEED, plots_dir)
        print(flush=True)

    print_experiment_metadata(metadata)
    print_summary(summary_df)
    print(f"Resultados salvos em {results_dir}", flush=True)
    print(f"Gráficos salvos em {plots_dir}", flush=True)
