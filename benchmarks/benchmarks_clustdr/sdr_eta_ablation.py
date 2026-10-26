import argparse
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch

from benchmarks.benchmarks_clustdr.classification_joint_benchmarks import (
    DATASET_LOADERS,
    PROJECT_ROOT,
    SDR_BASE_KWARGS,
    _compute_joint_metrics,
    _preprocess_features,
    _set_seed,
)
from src.sdr import SDR


DEFAULT_DATASETS = ("coil20", "snareseq")
DEFAULT_LAMBDAS_HSIC = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
DEFAULT_OUTPUT_SAMS = (20, 100, 180)
DEFAULT_SEEDS = (0, 1, 2)


def _make_sdr(
    seed: int,
    output_sam: int,
    output_dim: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
    lambda_hsic: float,
) -> SDR:
    sdr_kwargs = deepcopy(SDR_BASE_KWARGS)
    sdr_kwargs["lambda_hsic"] = lambda_hsic
    return SDR(
        **sdr_kwargs,
        output_sam=output_sam,
        output_dim=output_dim,
        seed=seed,
        device=device,
        dtype=torch.double,
        max_iter=max_iter,
        max_iter_outer=max_iter_outer,
        init_T="spectral",
    )


def _run_single_ablation(
    dataset_name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    output_sam: int,
    output_dim: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
    lambda_hsic: float,
) -> dict[str, float | int | str]:
    _set_seed(seed)
    method = _make_sdr(
        seed=seed,
        output_sam=output_sam,
        output_dim=output_dim,
        device=device,
        max_iter=max_iter,
        max_iter_outer=max_iter_outer,
        lambda_hsic=lambda_hsic,
    )

    start = torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None
    end = torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None

    if start is not None and end is not None:
        torch.cuda.synchronize()
        start.record()
        z, _ = method.fit_transform(x, y)
        end.record()
        torch.cuda.synchronize()
        runtime_s = start.elapsed_time(end) / 1000.0
    else:
        import time

        start_time = time.perf_counter()
        z, _ = method.fit_transform(x, y)
        runtime_s = time.perf_counter() - start_time

    metrics = _compute_joint_metrics(x=x, y=y, t=method.T.detach(), z=z.detach())
    return {
        "dataset": dataset_name,
        "method": "sdr",
        "seed": seed,
        "output_sam": output_sam,
        "output_dim": output_dim,
        "lambda_hsic": lambda_hsic,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_classes": int(torch.unique(y).numel()),
        "active_prototypes": int((method.T.sum(0) > 0).sum().item()),
        "homogeneity": metrics["homogeneity"],
        "nmi": metrics["nmi"],
        "silhouette": metrics["silhouette"],
        "kmeans": metrics["kmeans"],
        "fit_transform_runtime_s": runtime_s,
    }


def _aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby(
            ["dataset", "method", "lambda_hsic", "output_sam", "output_dim"],
            as_index=False,
        )
        .agg(
            homogeneity_mean=("homogeneity", "mean"),
            homogeneity_std=("homogeneity", "std"),
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            silhouette_mean=("silhouette", "mean"),
            silhouette_std=("silhouette", "std"),
            kmeans_mean=("kmeans", "mean"),
            kmeans_std=("kmeans", "std"),
            runtime_mean_s=("fit_transform_runtime_s", "mean"),
            runtime_std_s=("fit_transform_runtime_s", "std"),
            active_prototypes_mean=("active_prototypes", "mean"),
            active_prototypes_std=("active_prototypes", "std"),
        )
        .sort_values(["dataset", "lambda_hsic", "output_sam"])
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an SDR lambda_hsic ablation on the joint clustering/DR benchmarks "
            "for the COIL-20 and SNAREseq datasets."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--lambda-hsic", nargs="+", type=float, default=list(DEFAULT_LAMBDAS_HSIC))
    parser.add_argument("--output-sams", nargs="+", type=int, default=list(DEFAULT_OUTPUT_SAMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--max-iter-outer", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "benchmark_outputs" / "sdr_lambda_hsic_ablation"),
    )
    parser.add_argument("--coil20-path")
    parser.add_argument("--snareseq-path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    unknown_datasets = set(args.datasets) - set(DATASET_LOADERS)
    if unknown_datasets:
        raise ValueError(f"Unsupported datasets requested: {sorted(unknown_datasets)}")

    dataset_path_map = {
        "coil20": args.coil20_path,
        "snareseq": args.snareseq_path,
    }

    torch.set_default_dtype(torch.double)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cached_datasets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for dataset_name in args.datasets:
        if dataset_name not in dataset_path_map:
            raise ValueError(
                "This ablation runner is intended for coil20 and snareseq only. "
                f"Received dataset={dataset_name!r}."
            )
        x, y = DATASET_LOADERS[dataset_name](dataset_path_map[dataset_name])
        x = _preprocess_features(x)
        cached_datasets[dataset_name] = (x.to(args.device), y.to(args.device))

    rows: list[dict[str, float | int | str]] = []
    for dataset_name in args.datasets:
        x, y = cached_datasets[dataset_name]
        for lambda_hsic in args.lambda_hsic:
            for output_sam in args.output_sams:
                if output_sam > x.shape[0]:
                    continue
                for seed in args.seeds:
                    row = _run_single_ablation(
                        dataset_name=dataset_name,
                        x=x,
                        y=y,
                        seed=seed,
                        output_sam=output_sam,
                        output_dim=args.output_dim,
                        device=args.device,
                        max_iter=args.max_iter,
                        max_iter_outer=args.max_iter_outer,
                        lambda_hsic=lambda_hsic,
                    )
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                "dataset": dataset_name,
                                "method": "sdr",
                                "seed": seed,
                                "output_sam": output_sam,
                                "lambda_hsic": lambda_hsic,
                                "homogeneity": row["homogeneity"],
                                "nmi": row["nmi"],
                                "silhouette": row["silhouette"],
                                "kmeans": row["kmeans"],
                                "runtime_s": row["fit_transform_runtime_s"],
                            }
                        )
                    )

    results_df = pd.DataFrame(rows)
    summary_df = _aggregate_results(results_df)

    raw_path = results_dir / "run_metrics.csv"
    summary_path = results_dir / "summary.csv"
    config_path = results_dir / "config.json"

    results_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    config = {
        "datasets": args.datasets,
        "method": "sdr",
        "lambda_hsic": args.lambda_hsic,
        "output_sams": args.output_sams,
        "seeds": args.seeds,
        "output_dim": args.output_dim,
        "max_iter": args.max_iter,
        "max_iter_outer": args.max_iter_outer,
        "device": args.device,
    }
    config_path.write_text(json.dumps(config, indent=2))

    print(f"Saved raw run metrics to {raw_path}")
    print(f"Saved summary metrics to {summary_path}")
    print(f"Saved run configuration to {config_path}")


if __name__ == "__main__":
    main()
