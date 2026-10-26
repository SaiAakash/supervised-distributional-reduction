from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_RUNTIME_COLUMNS = {
    "DKL": ["dkl_train"],
    "DGP": ["deep_gp_train"],
    "KSPCA-GP": ["kernel_supervised_pca_fit", "kernel_supervised_pca_gp_train"],
    "NCA-GP": ["nca_fit", "nca_gp_train"],
    "UMAP-GP": ["supervised_umap_fit", "supervised_umap_gp_train"],
    "SDR-GP": ["sdr_fit", "sdrgp_train"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean and standard deviation of runtime columns method-wise "
            "across all *_runtimes.csv files in a benchmark-results folder."
        )
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        help=(
            "Directory containing runtime CSVs such as "
            "benchmark_outputs/DKL_vs_SDR_results/gp_regression_benchmarks "
            "or benchmark_outputs/DKL_vs_SDR_results/image_benchmarks_latest."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save the summary as a CSV file.",
    )
    return parser.parse_args()


def _collect_unified_training_rows(results_dir: Path) -> pd.DataFrame:
    runtime_paths = sorted(results_dir.glob("*_runtimes.csv"))
    if not runtime_paths:
        raise FileNotFoundError(f"No *_runtimes.csv files found in {results_dir}")

    frames: list[pd.DataFrame] = []
    for runtime_path in runtime_paths:
        df = pd.read_csv(runtime_path)
        if df.empty:
            continue

        base_columns = [column for column in ["dataset", "seed"] if column in df.columns]
        for model_name, runtime_columns in MODEL_RUNTIME_COLUMNS.items():
            available_columns = [column for column in runtime_columns if column in df.columns]
            if len(available_columns) != len(runtime_columns):
                continue

            model_df = df[base_columns].copy()
            model_df["model"] = model_name
            model_df["training_runtime_s"] = df[available_columns].sum(axis=1)
            frames.append(model_df)

    if not frames:
        raise ValueError(f"No runtime rows found in {results_dir}")

    return pd.concat(frames, ignore_index=True)


def summarize_runtimes(results_dir: Path) -> pd.DataFrame:
    runtimes_df = _collect_unified_training_rows(results_dir)
    summary_df = runtimes_df.groupby("model", as_index=False).agg(
        runtime_mean_s=("training_runtime_s", "mean"),
        runtime_std_s=("training_runtime_s", "std"),
        num_rows=("training_runtime_s", "count"),
    )
    return summary_df.sort_values("runtime_mean_s", ascending=False).reset_index(drop=True)


def summarize_runtimes_by_dataset(results_dir: Path) -> pd.DataFrame:
    runtimes_df = _collect_unified_training_rows(results_dir)
    if "dataset" not in runtimes_df.columns:
        raise ValueError("Runtime files do not contain a dataset column.")

    summary_df = runtimes_df.groupby(["dataset", "model"], as_index=False).agg(
        runtime_mean_s=("training_runtime_s", "mean"),
        runtime_std_s=("training_runtime_s", "std"),
        num_rows=("training_runtime_s", "count"),
    )
    return summary_df.sort_values(["dataset", "runtime_mean_s"], ascending=[True, False]).reset_index(
        drop=True
    )


def main() -> None:
    args = parse_args()
    summary_df = summarize_runtimes(args.results_dir)
    dataset_summary_df = summarize_runtimes_by_dataset(args.results_dir)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print("Overall summary\n")
        print(summary_df.to_string(index=False))
        print("\nDataset-wise summary\n")
        print(dataset_summary_df.to_string(index=False))

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.output_csv, index=False)
        dataset_output_csv = args.output_csv.with_name(
            f"{args.output_csv.stem}_by_dataset{args.output_csv.suffix}"
        )
        dataset_summary_df.to_csv(dataset_output_csv, index=False)
        print(f"\nSaved overall summary to {args.output_csv}")
        print(f"Saved dataset-wise summary to {dataset_output_csv}")


if __name__ == "__main__":
    main()
