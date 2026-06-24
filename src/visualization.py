from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DPI = 300
SCATTER_SAMPLE_SIZE = 10_000
SCATTER_SAMPLE_SEED = 0

MODEL_LINEAR = "linear_regression"
MODEL_RF = "random_forest"

MODEL_LABELS = {
    MODEL_LINEAR: "Regressão Linear",
    MODEL_RF: "Random Forest",
}


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_target_distribution(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["Global_active_power"], bins=50, color="steelblue", edgecolor="white")
    ax.set_title("Distribuição do Consumo de Energia (Global_active_power)")
    ax.set_xlabel("Potência ativa global (kW)")
    ax.set_ylabel("Frequência")
    _save_figure(fig, output_path)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
) -> None:
    columns = feature_columns + ["Global_active_power"]
    correlation = df[columns].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title("Matriz de Correlação")
    _save_figure(fig, output_path)


def plot_consumption_timeseries(df: pd.DataFrame, output_path: Path) -> None:
    hourly = (
        df.set_index("Datetime")["Global_active_power"]
        .resample("h")
        .mean()
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hourly.index, hourly.values, linewidth=0.8, color="steelblue")
    ax.set_title("Consumo de Energia ao Longo do Tempo (média horária)")
    ax.set_xlabel("Data")
    ax.set_ylabel("Potência ativa global (kW)")
    fig.autofmt_xdate()
    _save_figure(fig, output_path)


def plot_real_vs_predicted(
    y_test: np.ndarray,
    y_pred_linear: np.ndarray,
    y_pred_random_forest: np.ndarray,
    plot_seed: int,
    output_path: Path,
) -> None:
    rng = np.random.default_rng(SCATTER_SAMPLE_SEED)
    sample_size = min(SCATTER_SAMPLE_SIZE, len(y_test))
    indices = rng.choice(len(y_test), size=sample_size, replace=False)

    y_sample = y_test[indices]
    lr_sample = y_pred_linear[indices]
    rf_sample = y_pred_random_forest[indices]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        y_sample,
        lr_sample,
        alpha=0.4,
        s=10,
        label=MODEL_LABELS[MODEL_LINEAR],
        color="tab:blue",
    )
    ax.scatter(
        y_sample,
        rf_sample,
        alpha=0.4,
        s=10,
        label=MODEL_LABELS[MODEL_RF],
        color="tab:orange",
    )

    min_value = min(y_sample.min(), lr_sample.min(), rf_sample.min())
    max_value = max(y_sample.max(), lr_sample.max(), rf_sample.max())
    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        "k--",
        linewidth=1,
        label="y = x",
    )

    ax.set_title(f"Valores Reais vs Preditos (seed={plot_seed})")
    ax.set_xlabel("Valor real")
    ax.set_ylabel("Valor predito")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    _save_figure(fig, output_path)


def plot_metrics_boxplot(runs_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = runs_df.copy()
    plot_df["model"] = plot_df["model"].map(MODEL_LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("r2", "R²"),
    ]

    for ax, (column, title) in zip(axes, metrics):
        sns.boxplot(data=plot_df, x="model", y=column, ax=ax)
        ax.set_title(f"{title} — 30 execuções")
        ax.set_xlabel("")
        ax.set_ylabel(title)

    fig.suptitle("Distribuição das Métricas por Modelo")
    fig.tight_layout()
    _save_figure(fig, output_path)


def generate_exploratory_plots(
    featured_df: pd.DataFrame,
    feature_columns: list[str],
    plots_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    plot_target_distribution(
        featured_df,
        plots_dir / "target_distribution.png",
    )
    plot_correlation_heatmap(
        featured_df,
        feature_columns,
        plots_dir / "correlation_heatmap.png",
    )
    plot_consumption_timeseries(
        featured_df,
        plots_dir / "consumption_timeseries.png",
    )


def generate_result_plots(
    runs_df: pd.DataFrame,
    plot_predictions: dict[str, np.ndarray],
    plot_seed: int,
    plots_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    plot_real_vs_predicted(
        plot_predictions["y_test"],
        plot_predictions["y_pred_linear_regression"],
        plot_predictions["y_pred_random_forest"],
        plot_seed,
        plots_dir / "real_vs_predicted.png",
    )
    plot_metrics_boxplot(
        runs_df,
        plots_dir / "metrics_boxplot.png",
    )


def generate_all_plots(
    featured_df: pd.DataFrame,
    feature_columns: list[str],
    runs_df: pd.DataFrame,
    plot_predictions: dict[str, np.ndarray],
    plot_seed: int,
    plots_dir: Path,
) -> None:
    generate_exploratory_plots(featured_df, feature_columns, plots_dir)
    generate_result_plots(runs_df, plot_predictions, plot_seed, plots_dir)
