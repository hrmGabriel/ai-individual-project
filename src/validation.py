from pathlib import Path

import pandas as pd

from src.experiment_runner import MODEL_LINEAR, MODEL_RF, SEEDS

EXPECTED_PLOTS = [
    "target_distribution.png",
    "correlation_heatmap.png",
    "consumption_timeseries.png",
    "real_vs_predicted.png",
    "metrics_boxplot.png",
]

EXPECTED_CSVS = [
    "all_runs.csv",
    "linear_regression_runs.csv",
    "random_forest_runs.csv",
    "summary.csv",
]

RUN_COLUMNS = ["seed", "model", "mae", "rmse", "r2", "train_time", "total_time"]
SUMMARY_COLUMNS = ["model", "metric", "mean", "std", "median"]


def validate_experiment_outputs(plots_dir: Path, results_dir: Path) -> None:
    errors: list[str] = []

    for filename in EXPECTED_PLOTS:
        path = plots_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"Gráfico ausente ou vazio: {path}")

    for filename in EXPECTED_CSVS:
        path = results_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"CSV ausente ou vazio: {path}")

    metadata_path = results_dir / "experiment_metadata.json"
    if not metadata_path.exists():
        errors.append(f"Metadados ausentes: {metadata_path}")

    if not errors:
        all_runs = pd.read_csv(results_dir / "all_runs.csv")
        lr_runs = pd.read_csv(results_dir / "linear_regression_runs.csv")
        rf_runs = pd.read_csv(results_dir / "random_forest_runs.csv")
        summary = pd.read_csv(results_dir / "summary.csv")

        if len(all_runs) != len(SEEDS) * 2:
            errors.append(
                f"all_runs.csv esperava {len(SEEDS) * 2} linhas, "
                f"encontrou {len(all_runs)}"
            )

        if len(lr_runs) != len(SEEDS):
            errors.append(
                f"linear_regression_runs.csv esperava {len(SEEDS)} linhas, "
                f"encontrou {len(lr_runs)}"
            )

        if len(rf_runs) != len(SEEDS):
            errors.append(
                f"random_forest_runs.csv esperava {len(SEEDS)} linhas, "
                f"encontrou {len(rf_runs)}"
            )

        for column in RUN_COLUMNS:
            if column not in all_runs.columns:
                errors.append(f"Coluna ausente em all_runs.csv: {column}")

        for column in SUMMARY_COLUMNS:
            if column not in summary.columns:
                errors.append(f"Coluna ausente em summary.csv: {column}")

        if set(all_runs["model"].unique()) != {MODEL_LINEAR, MODEL_RF}:
            errors.append("Modelos inesperados em all_runs.csv")

        if set(all_runs["seed"]) != set(SEEDS):
            errors.append("Seeds inesperadas em all_runs.csv")

    if errors:
        raise ValueError("Validação falhou:\n- " + "\n- ".join(errors))

    print("Validação concluída: todos os artefatos foram gerados corretamente.")
