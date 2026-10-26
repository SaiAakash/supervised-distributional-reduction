"""
Hybrid sequential baselines: SDR for the DR stage, unsupervised SPECTRAL
clustering for the clustering stage.

Where classification_sdr_sequential / joint_vs_sequential use SDR for *both*
stages, this script swaps the clustering stage for the same unsupervised
spectral clustering that the DistR versions of Clust_then_DR / DR_then_Clust use
(sklearn SpectralClustering on a precomputed affinity, hard one-hot assignment).
This isolates whether SDR's supervised joint coupling helps the clustering step
specifically, versus plain spectral clustering glued to an SDR embedding.

    ==========================  ===============================================
    Method                      Pipeline
    ==========================  ===============================================
    clust_spectral_then_dr_sdr  Spectral clustering on P_X (unsupervised) ->
                                prototype barycentres -> SDR (pure DR) to reduce
                                prototype dimensions. T = T_spectral @ T_dr.
    dr_sdr_then_clust_spectral  SDR (pure DR) on all samples -> spectral
                                clustering on the embedding affinity P_Z
                                (unsupervised) -> prototype barycentres.
                                T = T_dr @ T_spectral.
    ==========================  ===============================================

The DR stage reuses the exact SDR affinities / kernels / hyperparameters from
method_comparison.py; the spectral stage mirrors src.clust_dr.Clust_then_DR /
DR_then_Clust (SpectralClustering with affinity="precomputed", n_init=10).

Logging/aggregation match joint_vs_sequential_benchmarks.py: per-run raw metrics,
mean/std across seeds, per-method CSV splits, config dump. Metrics are
Homogeneity, NMI, K-means NMI and Silhouette.
"""

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
from sklearn.cluster import SpectralClustering

from src.affinities import NormalizedGaussianAndStudentAffinity, SymmetricEntropicAffinity
from src.scores import scores_clustdr
from src.sdr import SDR
from src.utils import barycenter_feat, plan_color

from benchmarks.benchmarks_clustdr.classification_joint_benchmarks import _preprocess_features


DEFAULT_DATASETS = ("coil20", "snareseq")
DEFAULT_METHODS = ("clust_spectral_then_dr_sdr", "dr_sdr_then_clust_spectral")
DEFAULT_SEEDS = tuple(range(5))

# SDR config for the DR stage, identical to method_comparison.py's SDR family.
SDR_BASE_KWARGS = {
    "affinity_embedding": NormalizedGaussianAndStudentAffinity(student=True),
    "hsic_kernel_z": "rq",
    "hsic_kernel_y": "delta",
    "fgw_loss": "ce_loss",
    "loss_fun": "kl_loss",
    "objective": "sdr",
    "alpha": 0.2,
    "lambda_hsic": 1000.0,
    "optimizer": "Adam",
    "init": "normal",
    "tol": 1e-5,
    "verbose": False,
}

DATASET_AFFINITY_KWARGS = {
    "coil20": {"perp": 20, "lr": 0.1},
    "snareseq": {"perp": 30, "lr": 0.01},
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
    path = _resolve_data_path(path_str, "coil20_50d.pt")
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or "X" not in data or "y" not in data:
        raise ValueError(f"Expected a dict with X/y tensors at {path}.")
    return data["X"].to(torch.double), data["y"].to(torch.long)


def _load_snareseq(path_str: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = _load_from_feature_label_tensor(_resolve_data_path(path_str, "SNAREseq_atac.pt"))
    return x, y - y.min()


DATASET_LOADERS: dict[str, Callable[[str | None], tuple[torch.Tensor, torch.Tensor]]] = {
    "coil20": _load_coil20,
    "snareseq": _load_snareseq,
}


def _make_sdr(
    dataset_name: str,
    seed: int,
    output_sam: int,
    output_dim: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
    init_t: str | torch.Tensor,
) -> SDR:
    affinity_kwargs = DATASET_AFFINITY_KWARGS.get(dataset_name)
    if affinity_kwargs is None:
        raise ValueError(f"Unsupported dataset for affinity configuration: {dataset_name}")
    return SDR(
        **deepcopy(SDR_BASE_KWARGS),
        affinity_data=SymmetricEntropicAffinity(**affinity_kwargs, verbose=False),
        output_sam=output_sam,
        output_dim=output_dim,
        seed=seed,
        device=device,
        dtype=torch.double,
        max_iter=max_iter,
        max_iter_outer=max_iter_outer,
        init_T=init_t,
    )


def _identity_plan(n_samples: int, device: str) -> torch.Tensor:
    return torch.eye(n_samples, dtype=torch.double, device=device)


def _spectral_plan(
    affinity: torch.Tensor,
    output_sam: int,
    seed: int,
    device: str,
) -> torch.Tensor:
    """Unsupervised spectral clustering on a precomputed affinity -> one-hot plan.

    Mirrors src.clust_dr.Clust_then_DR / DR_then_Clust: shift to non-negative if
    needed, SpectralClustering(affinity="precomputed"), then a hard one-hot
    assignment matrix of shape (n_points, output_sam).
    """
    kernel = affinity.detach().clone()
    if (kernel < 0).any():
        kernel = kernel - kernel.min()
    labels = SpectralClustering(
        n_clusters=output_sam,
        affinity="precomputed",
        random_state=seed,
        n_init=10,
    ).fit(kernel.cpu().numpy()).labels_
    return torch.eye(output_sam)[labels].to(dtype=torch.double, device=device)


def _run_clust_spectral_then_dr_sdr(
    dataset_name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    output_sam: int,
    output_dim: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unsupervised spectral clustering, then SDR dimensionality reduction."""
    affinity_kwargs = DATASET_AFFINITY_KWARGS[dataset_name]
    affinity_data = SymmetricEntropicAffinity(**affinity_kwargs, verbose=False)
    p_x = affinity_data.compute_affinity(x).detach()
    t_spectral = _spectral_plan(p_x, output_sam, seed, device)

    x_proto = barycenter_feat(x, t_spectral)
    proto_y, _ = plan_color(t_spectral, y)

    dr_stage = _make_sdr(
        dataset_name=dataset_name,
        seed=seed,
        output_sam=output_sam,
        output_dim=output_dim,
        device=device,
        max_iter=max_iter,
        max_iter_outer=max_iter_outer,
        init_t=_identity_plan(output_sam, device),
    )
    z_final, _ = dr_stage.fit_transform(x_proto.detach(), proto_y.detach())

    t_final = t_spectral @ dr_stage.T.detach()
    return z_final.detach(), t_final.detach()


def _run_dr_sdr_then_clust_spectral(
    dataset_name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    output_sam: int,
    output_dim: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SDR dimensionality reduction, then unsupervised spectral clustering."""
    n_samples = int(x.shape[0])
    dr_stage = _make_sdr(
        dataset_name=dataset_name,
        seed=seed,
        output_sam=n_samples,
        output_dim=output_dim,
        device=device,
        max_iter=max_iter,
        max_iter_outer=max_iter_outer,
        init_t=_identity_plan(n_samples, device),
    )
    z_large, _ = dr_stage.fit_transform(x, y)
    z_large = z_large.detach()

    affinity_embedding = NormalizedGaussianAndStudentAffinity(student=True)
    p_z = affinity_embedding.compute_affinity(z_large).detach()
    t_spectral = _spectral_plan(p_z, output_sam, seed, device)

    z_final = barycenter_feat(z_large, t_spectral)
    t_final = dr_stage.T.detach() @ t_spectral
    return z_final.detach(), t_final.detach()


METHOD_RUNNERS = {
    "clust_spectral_then_dr_sdr": _run_clust_spectral_then_dr_sdr,
    "dr_sdr_then_clust_spectral": _run_dr_sdr_then_clust_spectral,
}


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

    runner = METHOD_RUNNERS.get(method_name)
    if runner is None:
        raise ValueError(f"Unknown method {method_name}")

    start = time.perf_counter()
    z, t = runner(
        dataset_name=dataset_name,
        x=x,
        y=y,
        seed=seed,
        output_sam=output_sam,
        output_dim=output_dim,
        device=device,
        max_iter=max_iter,
        max_iter_outer=max_iter_outer,
    )
    runtime_s = time.perf_counter() - start

    metrics = _compute_joint_metrics(x=x, y=y, t=t, z=z)
    return {
        "dataset": dataset_name,
        "method": method_name,
        "clustering": "spectral",
        "dr": "sdr",
        "seed": seed,
        "output_sam": output_sam,
        "output_dim": output_dim,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_classes": int(torch.unique(y).numel()),
        "active_prototypes": int((t.sum(0) > 0).sum().item()),
        "homogeneity": metrics["homogeneity"],
        "nmi": metrics["nmi"],
        "silhouette": metrics["silhouette"],
        "kmeans": metrics["kmeans"],
        "fit_transform_runtime_s": runtime_s,
    }


def _aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby(
            ["dataset", "method", "clustering", "dr", "output_sam", "output_dim"],
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
        .sort_values(["dataset", "method", "output_sam"])
    )


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid SDR-DR + spectral-clustering sequential baselines."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-sam", type=int, default=100)
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--max-iter-outer", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "benchmark_outputs" / "hybrid_spectral_sdr_sequential"),
    )
    parser.add_argument("--coil20-path")
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
        if args.output_sam > x.shape[0]:
            raise ValueError(
                f"output_sam={args.output_sam} exceeds sample count {x.shape[0]} for {dataset_name}."
            )
        for method_name in args.methods:
            for seed in args.seeds:
                row = _run_single_benchmark(
                    dataset_name=dataset_name,
                    method_name=method_name,
                    x=x,
                    y=y,
                    seed=seed,
                    output_sam=args.output_sam,
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
                            "output_sam": args.output_sam,
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
    per_method_paths = _write_method_specific_results(results_df, summary_df, results_dir)

    config = {
        "datasets": args.datasets,
        "methods": args.methods,
        "seeds": args.seeds,
        "output_sam": args.output_sam,
        "output_dim": args.output_dim,
        "max_iter": args.max_iter,
        "max_iter_outer": args.max_iter_outer,
        "device": args.device,
        "spectral_clustering": {"affinity": "precomputed", "n_init": 10},
        "sdr_base_kwargs": {k: str(v) for k, v in SDR_BASE_KWARGS.items()},
        "dataset_affinity_kwargs": DATASET_AFFINITY_KWARGS,
    }
    config_path.write_text(json.dumps(config, indent=2))

    print(f"Saved raw run metrics to {raw_path}")
    print(f"Saved summary metrics to {summary_path}")
    for method_name, method_raw_path, method_summary_path in per_method_paths:
        print(f"Saved {method_name} raw run metrics to {method_raw_path}")
        print(f"Saved {method_name} summary metrics to {method_summary_path}")


if __name__ == "__main__":
    main()
