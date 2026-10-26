import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.affinities import (
    NormalizedGaussianAndStudentAffinity,
    SymmetricEntropicAffinity,
)
from src.clust_dr import Clust_then_DR, DR_then_Clust, DistR
from src.scores import scores_clustdr
from src.sdr import SDR


DEFAULT_DATASETS = ("coil20", "fmnist", "snareseq")
DEFAULT_METHODS = ("clust_then_dr", "dr_then_clust", "distr", "sdr")
DEFAULT_SEEDS = tuple(range(5))
DEFAULT_OUTPUT_SAMS = tuple(range(20, 181, 40))

DATASET_AFFINITY_KWARGS = {
    "coil20": {"perp": 20, "lr": 0.1},
    "snareseq": {"perp": 30, "lr": 0.01},
    "fmnist": {"perp": 50, "lr": 0.01},
}

UNSUPERVISED_BASE_KWARGS = {
    "affinity_embedding": NormalizedGaussianAndStudentAffinity(student=True),
    "loss_fun": "kl_loss",
    "optimizer": "Adam",
    "init": "normal",
    "tol": 1e-5,
    "verbose": False,
}

SDR_BASE_KWARGS = {
    "affinity_embedding": NormalizedGaussianAndStudentAffinity(student=True),
    "hsic_kernel_z": "rq",
    "hsic_kernel_y": "delta",
    "fgw_loss": "ce_loss",
    "loss_fun": "kl_loss",
    "alpha": 0.2,
    "optimizer": "Adam",
    "init": "normal",
    "tol": 1e-5,
    "lambda_hsic": 1000.0,
    "verbose": False,
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_data_path(path_str: str | None, fallback_name: str) -> Path:
    candidates = []
    if path_str is not None:
        candidates.append(Path(path_str))
    candidates.extend(
        [
            PROJECT_ROOT / "data" / fallback_name,
            Path("data") / fallback_name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate dataset file for {fallback_name}.")


def _load_from_feature_label_tensor(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict) and "X" in data and "y" in data:
        return data["X"].to(torch.double), data["y"].to(torch.long)
    if isinstance(data, torch.Tensor) and data.ndim == 2:
        x = data[:, :-1].to(torch.double)
        y = data[:, -1].to(torch.long)
        return x, y
    raise ValueError(
        f"Expected either a dict with X/y tensors or a 2D tensor at {path}, found {type(data)}"
    )


def _load_coil20(path_str: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    path = _resolve_data_path(path_str, "coil20.pt")
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or "X" not in data or "y" not in data:
        raise ValueError(f"Expected a dict with X/y tensors at {path}.")
    return data["X"].to(torch.double), data["y"].to(torch.long)


def _load_fmnist(path_str: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    return _load_from_feature_label_tensor(_resolve_data_path(path_str, "fmnist.pt"))


def _load_snareseq(path_str: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = _load_from_feature_label_tensor(_resolve_data_path(path_str, "SNAREseq_atac.pt"))
    return x, y - y.min()


def _preprocess_features(x: torch.Tensor) -> torch.Tensor:
    # Standardize each input column and guard against zero-variance features.
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-12)
    return (x - mean) / std


DATASET_LOADERS: dict[str, Callable[[str | None], tuple[torch.Tensor, torch.Tensor]]] = {
    "coil20": _load_coil20,
    "fmnist": _load_fmnist,
    "snareseq": _load_snareseq,
}


def _make_method(
    dataset_name: str,
    method_name: str,
    seed: int,
    output_sam: int,
    output_dim: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
) -> object:
    affinity_kwargs = DATASET_AFFINITY_KWARGS.get(dataset_name)
    if affinity_kwargs is None:
        raise ValueError(f"Unsupported dataset for affinity configuration: {dataset_name}")

    common = {
        "output_sam": output_sam,
        "output_dim": output_dim,
        "seed": seed,
        "device": device,
        "dtype": torch.double,
        "max_iter": max_iter,
    }

    if method_name == "clust_then_dr":
        clust_then_dr_kwargs = deepcopy(UNSUPERVISED_BASE_KWARGS)
        clust_then_dr_kwargs["affinity_data"] = SymmetricEntropicAffinity(
            **affinity_kwargs,
            verbose=False,
        )
        return Clust_then_DR(
            **clust_then_dr_kwargs,
            **common,
            # init_T="kmeans",
            init_T="spectral",
        )
    if method_name == "dr_then_clust":
        dr_then_clust_kwargs = deepcopy(UNSUPERVISED_BASE_KWARGS)
        dr_then_clust_kwargs["affinity_data"] = SymmetricEntropicAffinity(
            **affinity_kwargs,
            verbose=False,
        )
        dr_then_clust_kwargs["init"] = "pca"
        return DR_then_Clust(
            **dr_then_clust_kwargs,
            **common,
            init_T="spectral",
        )
    if method_name == "distr":
        distr_kwargs = deepcopy(UNSUPERVISED_BASE_KWARGS)
        distr_kwargs["affinity_data"] = SymmetricEntropicAffinity(
            **affinity_kwargs,
            verbose=False,
        )
        return DistR(
            **distr_kwargs,
            **common,
            init_T="spectral",
            max_iter_outer=max_iter_outer,
        )
    if method_name == "sdr":
        sdr_kwargs = deepcopy(SDR_BASE_KWARGS)
        sdr_kwargs["affinity_data"] = SymmetricEntropicAffinity(
            **affinity_kwargs,
            verbose=False,
        )
        return SDR(
            **sdr_kwargs,
            **common,
            init_T="spectral",
            max_iter_outer=max_iter_outer,
        )
    raise ValueError(f"Unknown method {method_name}")


def _compute_joint_metrics(
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    z: torch.Tensor,
) -> dict[str, float]:
    scores = scores_clustdr(
        t,
        z,
        y,
        x,
        score_list=["hom", "nmi", "sil", "kmeans"],
    )
    return {
        "homogeneity": scores["hom"],
        "nmi": scores["nmi"],
        "silhouette": scores["sil"],
        "kmeans": scores["kmeans"],
    }


def _run_single_benchmark(
    dataset_name: str,
    method_name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    output_sam: int,
    output_dim: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
) -> dict[str, float | int | str]:
    _set_seed(seed)
    method = _make_method(
        dataset_name=dataset_name,
        method_name=method_name,
        seed=seed,
        output_sam=output_sam,
        output_dim=output_dim,
        device=device,
        max_iter=max_iter,
        max_iter_outer=max_iter_outer,
    )

    start = time.perf_counter()
    if method_name == "sdr":
        z, _ = method.fit_transform(x, y)
    else:
        z = method.fit_transform(x)
    runtime_s = time.perf_counter() - start

    metrics = _compute_joint_metrics(x=x, y=y, t=method.T.detach(), z=z.detach())
    return {
        "dataset": dataset_name,
        "method": method_name,
        "seed": seed,
        "output_sam": output_sam,
        "output_dim": output_dim,
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
    summary = (
        results_df.groupby(["dataset", "method", "output_sam", "output_dim"], as_index=False)
        .agg(
            homogeneity_mean=("homogeneity", "mean"),
            homogeneity_std=("homogeneity", "std"),
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            silhouette_mean=("silhouette", "mean"),
            silhouette_std=("silhouette", "std"),
            runtime_mean_s=("fit_transform_runtime_s", "mean"),
            runtime_std_s=("fit_transform_runtime_s", "std"),
            active_prototypes_mean=("active_prototypes", "mean"),
            kmeans_mean=("kmeans", "mean"),
            kmeans_std=("kmeans", "std"),
        )
        .sort_values(["dataset", "method", "output_sam"])
    )
    return summary


def _write_method_specific_results(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    results_dir: Path,
) -> list[tuple[str, Path, Path]]:
    written_paths: list[tuple[str, Path, Path]] = []
    for method_name in sorted(results_df["method"].unique()):
        method_results = results_df[results_df["method"] == method_name].copy()
        method_summary = summary_df[summary_df["method"] == method_name].copy()
        raw_path = results_dir / f"run_metrics_{method_name}.csv"
        summary_path = results_dir / f"summary_{method_name}.csv"
        method_results.to_csv(raw_path, index=False)
        method_summary.to_csv(summary_path, index=False)
        written_paths.append((method_name, raw_path, summary_path))
    return written_paths


def _write_dataset_results(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    results_root_dir: Path,
    config: dict[str, object],
) -> None:
    for dataset_name in sorted(results_df["dataset"].unique()):
        dataset_results_dir = results_root_dir / dataset_name
        dataset_results_dir.mkdir(parents=True, exist_ok=True)

        dataset_results = results_df[results_df["dataset"] == dataset_name].copy()
        dataset_summary = summary_df[summary_df["dataset"] == dataset_name].copy()

        raw_path = dataset_results_dir / "run_metrics.csv"
        summary_path = dataset_results_dir / "summary.csv"
        config_path = dataset_results_dir / "config.json"

        dataset_results.to_csv(raw_path, index=False)
        dataset_summary.to_csv(summary_path, index=False)
        per_method_paths = _write_method_specific_results(
            dataset_results,
            dataset_summary,
            dataset_results_dir,
        )

        dataset_config = dict(config)
        dataset_config["datasets"] = [dataset_name]
        config_path.write_text(json.dumps(dataset_config, indent=2))

        print(f"Saved raw run metrics to {raw_path}")
        print(f"Saved summary metrics to {summary_path}")
        for method_name, method_raw_path, method_summary_path in per_method_paths:
            print(f"Saved {method_name} raw run metrics to {method_raw_path}")
            print(f"Saved {method_name} summary metrics to {method_summary_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark joint clustering/DR methods on classification datasets."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-sams", nargs="+", type=int, default=list(DEFAULT_OUTPUT_SAMS))
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--max-iter-outer", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "benchmark_outputs" / "Joint_Clust_DR_results"),
    )
    parser.add_argument("--coil20-path")
    parser.add_argument("--fmnist-path")
    parser.add_argument("--snareseq-path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    unknown_datasets = set(args.datasets) - set(DATASET_LOADERS)
    if unknown_datasets:
        raise ValueError(f"Unsupported datasets requested: {sorted(unknown_datasets)}")
    unknown_methods = set(args.methods) - set(DEFAULT_METHODS)
    if unknown_methods:
        raise ValueError(f"Unsupported methods requested: {sorted(unknown_methods)}")

    dataset_path_map = {
        "coil20": args.coil20_path,
        "fmnist": args.fmnist_path,
        "snareseq": args.snareseq_path,
    }

    torch.set_default_dtype(torch.double)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cached_datasets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for dataset_name in args.datasets:
        x, y = DATASET_LOADERS[dataset_name](dataset_path_map[dataset_name])
        x = _preprocess_features(x)
        cached_datasets[dataset_name] = (x.to(args.device), y.to(args.device))

    rows: list[dict[str, float | int | str]] = []
    for dataset_name in args.datasets:
        x, y = cached_datasets[dataset_name]
        for method_name in args.methods:
            for output_sam in args.output_sams:
                if output_sam > x.shape[0]:
                    continue
                for seed in args.seeds:
                    row = _run_single_benchmark(
                        dataset_name=dataset_name,
                        method_name=method_name,
                        x=x,
                        y=y,
                        seed=seed,
                        output_sam=output_sam,
                        output_dim=args.output_dim,
                        device=args.device,
                        max_iter=args.max_iter,
                        max_iter_outer=args.max_iter_outer,
                    )
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                "dataset": dataset_name,
                                "method": method_name,
                                "seed": seed,
                                "output_sam": output_sam,
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

    config = {
        "datasets": args.datasets,
        "methods": args.methods,
        "seeds": args.seeds,
        "output_sams": args.output_sams,
        "output_dim": args.output_dim,
        "max_iter": args.max_iter,
        "max_iter_outer": args.max_iter_outer,
        "device": args.device,
    }
    _write_dataset_results(results_df, summary_df, results_dir, config)


if __name__ == "__main__":
    main()
