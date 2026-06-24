from pathlib import Path

from src.experiment_runner import run_experiment
from src.validation import validate_experiment_outputs


def main() -> None:
    root = Path(__file__).resolve().parent
    data_path = root / "data" / "household_power_consumption.txt"
    plots_dir = root / "plots"
    results_dir = root / "results"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {data_path}")

    plots_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    run_experiment(
        data_path=data_path,
        plots_dir=plots_dir,
        results_dir=results_dir,
    )
    validate_experiment_outputs(plots_dir, results_dir)


if __name__ == "__main__":
    main()
