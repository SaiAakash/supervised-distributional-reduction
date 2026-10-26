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
    ("homogeneity_mean", "homogeneity_std", "Homogeneity score"),
    ("kmeans_mean", "kmeans_std", "k-Means NMI"),
    ("silhouette_mean", "silhouette_std", "Silhouette score"),
]

TITLE_FONT_SIZE = 40
AXIS_LABEL_FONT_SIZE = 35
TICK_FONT_SIZE = 30
LEGEND_FONT_SIZE = 30

LINE_WIDTH = 2.6
MARKER_SIZE = 7
GRID_WIDTH = 0.8

BASELINE_METHODS = {
    "clust_then_dr": {"label": "Clust -> DR", "color": "#1f77b4", "linestyle": "--"},
    "dr_then_clust": {"label": "DR -> Clust", "color": "#ff7f0e", "linestyle": "-."},
    "distr": {"label": "DistR", "color": "#2ca02c", "linestyle": ":"},
}

SDR_STYLE = {
    "label": "SDR",
    "color": "#d62728",
    "marker": "o",
}

DATASET_DIR_ALIASES = {
    "coil20": ["coil20", "coil-20"],
    "coil-20": ["coil-20", "coil20"],
    "fmnist": ["fmnist"],
    "snareseq": ["snareseq"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a combined 3x3 eta-ablation figure with metric rows and "
            "prototype-count columns."
        )
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("benchmark_outputs/eta_ablation_coil20/summary.csv"),
        help="Path to the eta-ablation summary.csv file.",
    )
    parser.add_argument(
        "--prototype-counts",
        nargs=3,
        type=int,
        default=[20, 100, 180],
        help="Exactly three prototype counts to use as the figure columns.",
    )
    parser.add_argument(
        "--output-subdir",
        default="eta_plots",
        help="Subdirectory created beside summary.csv for the output plot.",
    )
    parser.add_argument(
        "--output-name",
        default="eta_ablation_metrics_grid.pdf",
        help="Filename for the combined 3x3 figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved figure DPI.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help=(
            "Optional dataset-specific directory containing summary_<method>.csv files "
            "for the flat baseline lines."
        ),
    )
    parser.add_argument(
        "--modal-results-root",
        type=Path,
        default=Path("benchmark_outputs/Joint_Clust_DR_results/"),
        help=(
            "Root directory containing per-dataset summary_<method>.csv modal benchmark "
            "files when --baseline-dir is not provided."
        ),
    )
    return parser.parse_args()


def _load_baselines(
    baseline_dir: Path | None,
    modal_results_root: Path,
    dataset_name: str,
    output_sam: int,
    mean_column: str,
) -> list[dict[str, str | float]]:
    baseline_rows: list[dict[str, str | float]] = []
    dataset_dir = baseline_dir
    if dataset_dir is None:
        dataset_dir_candidates = DATASET_DIR_ALIASES.get(dataset_name.lower(), [dataset_name])
        for dataset_dir_name in dataset_dir_candidates:
            candidate = modal_results_root / dataset_dir_name
            if candidate.exists():
                dataset_dir = candidate
                break

    if dataset_dir is None:
        return baseline_rows

    for method_name, style in BASELINE_METHODS.items():
        summary_path = dataset_dir / f"summary_{method_name}.csv"
        if not summary_path.exists():
            continue

        method_df = pd.read_csv(summary_path)
        if mean_column not in method_df.columns:
            continue

        row = method_df[method_df["output_sam"] == output_sam]
        if row.empty:
            continue

        baseline_rows.append(
            {
                "label": style["label"],
                "color": style["color"],
                "linestyle": style["linestyle"],
                "value": float(row.iloc[0][mean_column]),
            }
        )

    return baseline_rows


def _plot_metric_on_axis(
    ax: plt.Axes,
    df: pd.DataFrame,
    mean_column: str,
    std_column: str,
    baseline_rows: list[dict[str, str | float]],
) -> None:
    df = df.sort_values("lambda_hsic")
    x_values = df["lambda_hsic"]

    ax.plot(
        x_values,
        df[mean_column],
        color=SDR_STYLE["color"],
        marker=SDR_STYLE["marker"],
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label=SDR_STYLE["label"],
    )

    if std_column in df.columns:
        lower = df[mean_column] - df[std_column]
        upper = df[mean_column] + df[std_column]
        ax.fill_between(
            x_values,
            lower,
            upper,
            color=SDR_STYLE["color"],
            alpha=0.14,
        )

    for baseline in baseline_rows:
        ax.hlines(
            y=baseline["value"],
            xmin=x_values.min(),
            xmax=x_values.max(),
            colors=baseline["color"],
            linestyles=baseline["linestyle"],
            linewidth=2.2,
            label=baseline["label"],
        )

    ax.set_xscale("log")
    ax.tick_params(axis="both", which="both", labelsize=TICK_FONT_SIZE)
    ax.grid(True, linestyle="--", linewidth=GRID_WIDTH, alpha=0.45)


def plot_summary_grid(
    summary_path: Path,
    prototype_counts: list[int],
    output_subdir: str,
    output_name: str,
    dpi: int,
    baseline_dir: Path | None,
    modal_results_root: Path,
) -> Path:
    df = pd.read_csv(summary_path)
    required_columns = {"lambda_hsic", "output_sam", "dataset"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{summary_path} is missing required columns: {sorted(missing_columns)}. "
            f"Available columns: {list(df.columns)}"
        )

    for mean_column, _, _ in METRICS:
        if mean_column not in df.columns:
            raise ValueError(
                f"{summary_path} is missing required metric column '{mean_column}'. "
                f"Available columns: {list(df.columns)}"
            )

    available_prototypes = set(df["output_sam"].astype(int).unique())
    missing_prototypes = [count for count in prototype_counts if count not in available_prototypes]
    if missing_prototypes:
        raise ValueError(
            f"{summary_path} does not contain rows for prototype counts {missing_prototypes}. "
            f"Available prototype counts: {sorted(available_prototypes)}"
        )

    output_dir = summary_path.parent / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    dataset_name = str(df["dataset"].iloc[0])
    fig, axes = plt.subplots(3, 3, figsize=(24, 22), sharex=False)

    for row_idx, (mean_column, std_column, metric_label) in enumerate(METRICS):
        for col_idx, prototype_count in enumerate(prototype_counts):
            ax = axes[row_idx, col_idx]
            prototype_df = df[df["output_sam"].astype(int) == int(prototype_count)].copy()
            baseline_rows = _load_baselines(
                baseline_dir=baseline_dir,
                modal_results_root=modal_results_root,
                dataset_name=dataset_name,
                output_sam=int(prototype_count),
                mean_column=mean_column,
            )

            _plot_metric_on_axis(
                ax=ax,
                df=prototype_df,
                mean_column=mean_column,
                std_column=std_column,
                baseline_rows=baseline_rows,
            )

            if row_idx == 0:
                ax.set_title(
                    f"{int(prototype_count)} prototypes",
                    fontsize=TITLE_FONT_SIZE,
                    pad=16,
                )

            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=AXIS_LABEL_FONT_SIZE)

            if row_idx == len(METRICS) - 1:
                ax.set_xlabel(r"$\eta$", fontsize=AXIS_LABEL_FONT_SIZE)

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
    return output_path


def main() -> None:
    args = parse_args()
    output_path = plot_summary_grid(
        summary_path=args.summary_path,
        prototype_counts=[int(count) for count in args.prototype_counts],
        output_subdir=args.output_subdir,
        output_name=args.output_name,
        dpi=args.dpi,
        baseline_dir=args.baseline_dir,
        modal_results_root=args.modal_results_root,
    )
    print(f"Saved eta ablation grid to {output_path}")


if __name__ == "__main__":
    main()
