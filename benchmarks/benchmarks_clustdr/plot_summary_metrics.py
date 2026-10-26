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


SUMMARY_METRICS = [
    ("homogeneity_mean", "homogeneity_std", "Homogeneity score"),
    ("kmeans_mean", "kmeans_std", "k-Means NMI"),
    ("silhouette_mean", "silhouette_std", "Silhouette score"),
]

RUNTIME_METRIC = ("runtime_mean_s", "runtime_std_s", "Runtime (s)")

TITLE_FONT_SIZE = 40
AXIS_LABEL_FONT_SIZE = 35
TICK_FONT_SIZE = 30
LEGEND_FONT_SIZE = 30

LINE_WIDTH = 2.6
MARKER_SIZE = 7
GRID_WIDTH = 0.8

METHOD_LABELS = {
    "clust_then_dr": "Clust -> DR",
    "dr_then_clust": "DR -> Clust",
    "distr": "DistR",
    "sdr": "SDR",
}

METHOD_COLORS = {
    "clust_then_dr": "#1f77b4",
    "dr_then_clust": "#ff7f0e",
    "distr": "#2ca02c",
    "sdr": "#d62728",
}

METHOD_MARKERS = {
    "clust_then_dr": "o",
    "dr_then_clust": "s",
    "distr": "^",
    "sdr": "D",
}

DATASET_ORDER = ["coil20", "fmnist", "snareseq"]
DATASET_TITLES = {
    "coil20": "COIL-20",
    "coil-20": "COIL-20",
    "fmnist": "Fashion-MNIST",
    "snareseq": "SNAREseq",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paper-style combined summary and runtime figures for modal "
            "benchmark results from COIL-20, Fashion-MNIST, and SNAREseq."
        )
    )
    parser.add_argument(
        "--coil20-dir",
        type=Path,
        default=Path("benchmark_outputs/Joint_Clust_DR_results/coil20"),
        help="Directory containing summary_<method>.csv files for COIL-20.",
    )
    parser.add_argument(
        "--fmnist-dir",
        type=Path,
        default=Path("benchmark_outputs/Joint_Clust_DR_results/fmnist"),
        help="Directory containing summary_<method>.csv files for Fashion-MNIST.",
    )
    parser.add_argument(
        "--snareseq-dir",
        type=Path,
        default=Path("benchmark_outputs/Joint_Clust_DR_results/snareseq"),
        help="Directory containing summary_<method>.csv files for SNAREseq.",
    )
    parser.add_argument(
        "--x-column",
        default="output_sam",
        help="Column to use on the x-axis. Defaults to output_sam.",
    )
    parser.add_argument(
        "--summary-glob",
        default="summary_*.csv",
        help="Glob pattern used to find per-method summary files inside each dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs/Joint_Clust_DR_results/combined_plots"),
        help="Directory where the combined figures will be saved.",
    )
    parser.add_argument(
        "--summary-output-name",
        default="summary_metrics_grid.pdf",
        help="Filename for the 3x3 summary figure.",
    )
    parser.add_argument(
        "--runtime-output-name",
        default="runtime_grid.pdf",
        help="Filename for the 1x3 runtime figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved figure DPI.",
    )
    return parser.parse_args()


def _format_dataset_title(dataset_name: str) -> str:
    return DATASET_TITLES.get(dataset_name.lower(), dataset_name)


def _load_dataset_summary(summary_dir: Path, summary_glob: str) -> pd.DataFrame:
    summary_paths = [
        path for path in sorted(summary_dir.glob(summary_glob)) if path.name != "summary.csv"
    ]
    if not summary_paths:
        raise FileNotFoundError(f"No {summary_glob} files found in {summary_dir}")

    frames: list[pd.DataFrame] = []
    for summary_path in summary_paths:
        df = pd.read_csv(summary_path)
        if "method" not in df.columns:
            df["method"] = summary_path.stem.removeprefix("summary_")
        frames.append(df)

    combined_df = pd.concat(frames, ignore_index=True)
    return combined_df.sort_values(["dataset", "method", "output_sam", "output_dim"])


def _plot_metric_on_axis(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_column: str,
    mean_column: str,
    std_column: str,
    *,
    log_y: bool = False,
) -> None:
    method_order = sorted(df["method"].unique())
    for method in method_order:
        method_df = df[df["method"] == method].sort_values(x_column)
        label = METHOD_LABELS.get(method, method.replace("_", " ").title())
        color = METHOD_COLORS.get(method)
        marker = METHOD_MARKERS.get(method, "o")

        ax.plot(
            method_df[x_column],
            method_df[mean_column],
            label=label,
            color=color,
            marker=marker,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
        )

        if std_column in method_df.columns:
            lower = method_df[mean_column] - method_df[std_column]
            upper = method_df[mean_column] + method_df[std_column]
            if log_y:
                lower = lower.clip(lower=1e-12)
            ax.fill_between(
                method_df[x_column],
                lower,
                upper,
                color=color,
                alpha=0.12,
            )

    if log_y:
        if (df[mean_column] <= 0).any():
            raise ValueError(
                f"Cannot use log scale for {mean_column} because it contains non-positive values."
            )
        ax.set_yscale("log")

    ax.tick_params(axis="both", which="both", labelsize=TICK_FONT_SIZE)
    ax.grid(True, linestyle="--", linewidth=GRID_WIDTH, alpha=0.45)


def _validate_dataframe(
    df: pd.DataFrame, x_column: str, required_metrics: list[tuple[str, str, str]]
) -> None:
    required_columns = {x_column, "method"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}. Available columns: {list(df.columns)}"
        )

    for mean_column, _, _ in required_metrics:
        if mean_column not in df.columns:
            raise ValueError(
                f"Missing required metric column '{mean_column}'. Available columns: {list(df.columns)}"
            )


def plot_summary_grid(
    dataset_frames: dict[str, pd.DataFrame],
    output_path: Path,
    x_column: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(24, 22), sharex=False)

    for row_idx, (mean_column, std_column, metric_label) in enumerate(SUMMARY_METRICS):
        for col_idx, dataset_key in enumerate(DATASET_ORDER):
            ax = axes[row_idx, col_idx]
            df = dataset_frames[dataset_key]

            _plot_metric_on_axis(
                ax=ax,
                df=df,
                x_column=x_column,
                mean_column=mean_column,
                std_column=std_column,
            )

            if row_idx == 0:
                dataset_name = (
                    str(df["dataset"].iloc[0]) if "dataset" in df.columns else dataset_key
                )
                ax.set_title(
                    _format_dataset_title(dataset_name),
                    fontsize=TITLE_FONT_SIZE,
                    pad=16,
                )

            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=AXIS_LABEL_FONT_SIZE)

            if row_idx == len(SUMMARY_METRICS) - 1:
                ax.set_xlabel("Number of Prototypes", fontsize=AXIS_LABEL_FONT_SIZE)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.94])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_figure(
    dataset_frames: dict[str, pd.DataFrame],
    output_path: Path,
    x_column: str,
    dpi: int,
) -> None:
    mean_column, std_column, metric_label = RUNTIME_METRIC
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5), sharex=False)

    for col_idx, dataset_key in enumerate(DATASET_ORDER):
        ax = axes[col_idx]
        df = dataset_frames[dataset_key]

        _plot_metric_on_axis(
            ax=ax,
            df=df,
            x_column=x_column,
            mean_column=mean_column,
            std_column=std_column,
            log_y=True,
        )

        dataset_name = str(df["dataset"].iloc[0]) if "dataset" in df.columns else dataset_key
        ax.set_title(
            _format_dataset_title(dataset_name),
            fontsize=TITLE_FONT_SIZE,
            pad=16,
        )
        ax.set_xlabel("Number of Prototypes", fontsize=AXIS_LABEL_FONT_SIZE)
        if col_idx == 0:
            ax.set_ylabel(metric_label, fontsize=AXIS_LABEL_FONT_SIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.90])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    dataset_dirs = {
        "coil20": args.coil20_dir,
        "fmnist": args.fmnist_dir,
        "snareseq": args.snareseq_dir,
    }

    dataset_frames = {
        dataset_key: _load_dataset_summary(summary_dir, args.summary_glob)
        for dataset_key, summary_dir in dataset_dirs.items()
    }

    for dataset_key, df in dataset_frames.items():
        _validate_dataframe(
            df,
            args.x_column,
            required_metrics=SUMMARY_METRICS + [RUNTIME_METRIC],
        )
        if df.empty:
            raise ValueError(f"{dataset_dirs[dataset_key]} did not contain any summary rows.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_output_path = args.output_dir / args.summary_output_name
    runtime_output_path = args.output_dir / args.runtime_output_name

    plot_summary_grid(
        dataset_frames=dataset_frames,
        output_path=summary_output_path,
        x_column=args.x_column,
        dpi=args.dpi,
    )
    print(f"Saved summary grid to {summary_output_path}")

    plot_runtime_figure(
        dataset_frames=dataset_frames,
        output_path=runtime_output_path,
        x_column=args.x_column,
        dpi=args.dpi,
    )
    print(f"Saved runtime grid to {runtime_output_path}")


if __name__ == "__main__":
    main()
