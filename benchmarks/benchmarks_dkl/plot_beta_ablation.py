from __future__ import annotations

import argparse
import os
from pathlib import Path

MPL_CONFIG_DIR = Path(".matplotlib")
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("train_mse_mean", "Train MSE"),
    ("test_mse_mean", "Test MSE"),
    ("train_mean_log_likelihood_mean", "Train MLL"),
    ("test_mean_log_likelihood_mean", "Test MLL"),
]

DATASET_ORDER = ["boston", "energy", "concrete"]
DATASET_TITLES = {
    "boston": "Boston",
    "energy": "Energy",
    "concrete": "Concrete",
}

TITLE_FONT_SIZE = 40
AXIS_LABEL_FONT_SIZE = 35
TICK_FONT_SIZE = 30
LEGEND_FONT_SIZE = 30

LINE_WIDTH = 2.6
MARKER_SIZE = 7
GRID_WIDTH = 0.8

SDR_METHOD_NAME = "sdr"
SDR_LABEL = "SDR-GP"
SDR_COLOR = "#d62728"

BASELINE_METHOD_LABELS = {
    "dkl": "DKL",
    "deep_gp": "DGP",
    "nca_gp": "NCA-GP",
    "supervised_umap_gp": "UMAP-GP",
    "kernel_supervised_pca_gp": "KSPCA-GP",
}

BASELINE_METHOD_ORDER = [
    "kernel_supervised_pca_gp",
    "nca_gp",
    "supervised_umap_gp",
    "dkl",
    "deep_gp",
]

BASELINE_METHOD_STYLES = {
    "kernel_supervised_pca_gp": {"color": "#1f77b4", "linestyle": (0, (3, 1, 1, 1))},
    "nca_gp": {"color": "#8c564b", "linestyle": "-."},
    "supervised_umap_gp": {"color": "#ff7f0e", "linestyle": (0, (5, 1))},
    "deep_gp": {"color": "#2ca02c", "linestyle": ":"},
    "dkl": {"color": "#9467bd", "linestyle": "--"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a combined 4x3 regression benchmark figure comparing SDR beta sweeps "
            "against flat baseline methods."
        )
    )
    parser.add_argument(
        "--sdr-dir",
        type=Path,
        default=Path("benchmark_outputs/sdr_beta_benchmarks"),
        help="Directory containing <dataset>_metrics_summary.csv SDR beta-sweep files.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("benchmark_outputs/DKL_vs_SDR_results/gp_regression_benchmarks"),
        help="Directory containing <dataset>_<method>_metrics_summary.csv baseline files.",
    )
    parser.add_argument(
        "--datasets",
        nargs=3,
        default=DATASET_ORDER,
        help="Exactly three dataset names to use for the grid columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs/sdr_beta_benchmarks/plots"),
        help="Directory where the combined regression figure will be written.",
    )
    parser.add_argument(
        "--output-name",
        default="sdr_beta_grid.pdf",
        help="Filename for the combined 4x3 figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved figure DPI.",
    )
    return parser.parse_args()


def _format_dataset_title(dataset_name: str) -> str:
    return DATASET_TITLES.get(dataset_name.lower(), dataset_name.replace("_", " ").title())


def _load_sdr_summary(sdr_dir: Path, dataset: str) -> pd.DataFrame:
    summary_path = sdr_dir / f"{dataset}_metrics_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing SDR summary file: {summary_path}")

    df = pd.read_csv(summary_path)
    if df.empty:
        raise ValueError(f"{summary_path} is empty.")
    if "beta" not in df.columns:
        raise ValueError(f"{summary_path} does not contain a beta column.")

    return df.sort_values("beta").reset_index(drop=True)


def _load_baseline_summaries(baseline_dir: Path, dataset: str) -> dict[str, pd.DataFrame]:
    baseline_frames: dict[str, pd.DataFrame] = {}
    for method_name in BASELINE_METHOD_ORDER:
        summary_path = baseline_dir / f"{dataset}_{method_name}_metrics_summary.csv"
        if not summary_path.exists():
            continue

        df = pd.read_csv(summary_path)
        if df.empty:
            continue
        baseline_frames[method_name] = df

    return baseline_frames


def _validate_sdr_metrics(df: pd.DataFrame) -> None:
    required_columns = {"beta"} | {metric_column for metric_column, _ in METRICS}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}. Available columns: {list(df.columns)}"
        )


def _plot_metric_on_axis(
    ax: plt.Axes,
    sdr_df: pd.DataFrame,
    baseline_frames: dict[str, pd.DataFrame],
    metric_column: str,
) -> None:
    ax.plot(
        sdr_df["beta"],
        sdr_df[metric_column],
        color=SDR_COLOR,
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label=SDR_LABEL,
    )

    for method_name, baseline_df in baseline_frames.items():
        if metric_column not in baseline_df.columns:
            continue
        value = float(baseline_df.iloc[0][metric_column])
        style = BASELINE_METHOD_STYLES.get(method_name, {"color": "#7f7f7f", "linestyle": "--"})
        label = BASELINE_METHOD_LABELS.get(method_name, method_name.replace("_", " ").title())
        ax.axhline(
            y=value,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            label=label,
        )

    ax.tick_params(axis="both", which="both", labelsize=TICK_FONT_SIZE)
    ax.grid(True, linestyle="--", linewidth=GRID_WIDTH, alpha=0.45)


def plot_regression_grid(
    sdr_dir: Path,
    baseline_dir: Path,
    datasets: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(len(METRICS), len(datasets), figsize=(24, 24), sharex=False)

    for row_idx, (metric_column, metric_label) in enumerate(METRICS):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            sdr_df = _load_sdr_summary(sdr_dir, dataset)
            _validate_sdr_metrics(sdr_df)
            baseline_frames = _load_baseline_summaries(baseline_dir, dataset)

            _plot_metric_on_axis(
                ax=ax,
                sdr_df=sdr_df,
                baseline_frames=baseline_frames,
                metric_column=metric_column,
            )

            if row_idx == 0:
                ax.set_title(_format_dataset_title(dataset), fontsize=TITLE_FONT_SIZE, pad=16)

            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=AXIS_LABEL_FONT_SIZE)

            if row_idx == len(METRICS) - 1:
                ax.set_xlabel(r"$\beta$", fontsize=AXIS_LABEL_FONT_SIZE)

    handles, labels = axes[0][0].get_legend_handles_labels()
    unique_handles: list[object] = []
    unique_labels: list[str] = []
    seen_labels: set[str] = set()
    for handle, label in zip(handles, labels):
        if label in seen_labels:
            continue
        seen_labels.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)

    fig.legend(
        unique_handles,
        unique_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=min(6, len(unique_labels)),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.93])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    datasets = [dataset.lower() for dataset in args.datasets]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output_dir / args.output_name
    plot_regression_grid(
        sdr_dir=args.sdr_dir,
        baseline_dir=args.baseline_dir,
        datasets=datasets,
        output_path=output_path,
        dpi=args.dpi,
    )
    print(f"Saved regression beta grid to {output_path}")


if __name__ == "__main__":
    main()
