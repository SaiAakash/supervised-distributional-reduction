"""
Ablate the SDR interpolation parameter alpha on COIL-20 and SNAREseq.

For objective="sdr", alpha plays two roles:

  * it weights the structural (GW/DR) term against the CKA supervision term in
    the embedding objective,
        L = alpha * structural_loss - lambda_hsic * supervision_loss
  * because alpha < 1, the coupling T is updated with fused Gromov-Wasserstein,
    where alpha trades the structural cost against the linear label cost.

So alpha -> 1 pushes SDR towards purely structural (unsupervised-coupling)
behaviour and alpha -> 0 towards label-dominated behaviour.

Every hyperparameter other than alpha is imported from method_comparison so the
sweep is directly comparable to the joint/sequential benchmarks (same
affinities, loss, HSIC kernels, spectral T initialisation, lambda_hsic and
per-dataset perplexity/learning rate).
"""

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from src.affinities import SymmetricEntropicAffinity
from src.sdr import SDR

from benchmarks.benchmarks_clustdr.classification_joint_benchmarks import _preprocess_features
from benchmarks.benchmarks_rebuttal.method_comparison import (
    BASE_AFFINITY_KWARGS,
    DATASET_AFFINITY_KWARGS,
    DATASET_LOADERS,
    FGW_LOSS_KWARGS,
    HSIC_KWARGS,
    _compute_cka_score,
    _compute_joint_metrics,
    _final_loss_terms,
    _set_seed,
)


DEFAULT_DATASETS = ("coil20", "snareseq")
DEFAULT_ALPHAS = (0.1, 0.25, 0.5, 0.75, 0.9)
DEFAULT_SEEDS = (0,)


def _make_sdr(
    dataset_name: str,
    seed: int,
    output_sam: int,
    output_dim: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
    alpha: float,
    lambda_hsic: float,
) -> SDR:
    affinity_kwargs = DATASET_AFFINITY_KWARGS.get(dataset_name)
    if affinity_kwargs is None:
        raise ValueError(f"Unsupported dataset for affinity configuration: {dataset_name}")

    return SDR(
        **deepcopy(BASE_AFFINITY_KWARGS),
        **deepcopy(HSIC_KWARGS),
        **deepcopy(FGW_LOSS_KWARGS),
        affinity_data=SymmetricEntropicAffinity(**affinity_kwargs, verbose=False),
        objective="sdr",
        seed=seed,
        output_sam=output_sam,
        output_dim=output_dim,
        device=device,
        dtype=torch.double,
        max_iter=max_iter,
        max_iter_outer=max_iter_outer,
        alpha=alpha,
        lambda_hsic=lambda_hsic,
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
    alpha: float,
    lambda_hsic: float,
) -> dict[str, float | int | str]:
    _set_seed(seed)
    method = _make_sdr(
        dataset_name=dataset_name,
        seed=seed,
        output_sam=output_sam,
        output_dim=output_dim,
        device=device,
        max_iter=max_iter,
        max_iter_outer=max_iter_outer,
        alpha=alpha,
        lambda_hsic=lambda_hsic,
    )

    start = time.perf_counter()
    z, _ = method.fit_transform(x, y)
    runtime_s = time.perf_counter() - start

    metrics = _compute_joint_metrics(x=x, y=y, t=method.T.detach(), z=z.detach())
    cka_score = _compute_cka_score(method=method, y=y, z=z)
    losses = _final_loss_terms(method)

    return {
        "dataset": dataset_name,
        "method": "sdr",
        "seed": seed,
        "output_sam": output_sam,
        "output_dim": output_dim,
        "alpha": alpha,
        "lambda_hsic": lambda_hsic,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_classes": int(torch.unique(y).numel()),
        "active_prototypes": int((method.T.sum(0) > 0).sum().item()),
        "homogeneity": metrics["homogeneity"],
        "nmi": metrics["nmi"],
        "silhouette": metrics["silhouette"],
        "kmeans": metrics["kmeans"],
        "cka_score": cka_score,
        "final_objective": losses["final_objective"],
        "final_structural_loss": losses["final_structural_loss"],
        "final_supervision_loss": losses["final_supervision_loss"],
        "fit_transform_runtime_s": runtime_s,
        "inner_steps": len(getattr(method, "inner_loss_history", []) or []),
        "outer_steps": len(getattr(method, "outer_loss_history", []) or []),
    }


def _aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby(
            ["dataset", "method", "alpha", "output_sam", "output_dim"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "count"),
            homogeneity_mean=("homogeneity", "mean"),
            homogeneity_std=("homogeneity", "std"),
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            silhouette_mean=("silhouette", "mean"),
            silhouette_std=("silhouette", "std"),
            kmeans_mean=("kmeans", "mean"),
            kmeans_std=("kmeans", "std"),
            cka_score_mean=("cka_score", "mean"),
            cka_score_std=("cka_score", "std"),
            final_objective_mean=("final_objective", "mean"),
            final_structural_loss_mean=("final_structural_loss", "mean"),
            final_supervision_loss_mean=("final_supervision_loss", "mean"),
            runtime_mean_s=("fit_transform_runtime_s", "mean"),
            active_prototypes_mean=("active_prototypes", "mean"),
        )
        .sort_values(["dataset", "alpha", "output_sam"])
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablate the SDR alpha parameter on COIL-20 and SNAREseq."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-sam", type=int, default=100)
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--lambda-hsic", type=float, default=1000.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--max-iter-outer", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "benchmark_outputs" / "sdr_alpha_ablation"),
    )
    parser.add_argument("--coil20-path")
    parser.add_argument("--snareseq-path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    unknown_datasets = set(args.datasets) - set(DATASET_LOADERS)
    if unknown_datasets:
        raise ValueError(f"Unsupported datasets requested: {sorted(unknown_datasets)}")

    invalid_alphas = [a for a in args.alphas if not 0.0 < a < 1.0]
    if invalid_alphas:
        raise ValueError(
            f"alpha must satisfy 0 < alpha < 1 for the FGW/SDR objective; got {invalid_alphas}."
        )

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

        for alpha in args.alphas:
            for seed in args.seeds:
                row = _run_single_ablation(
                    dataset_name=dataset_name,
                    x=x,
                    y=y,
                    seed=seed,
                    output_sam=args.output_sam,
                    output_dim=args.output_dim,
                    device=args.device,
                    max_iter=args.max_iter,
                    max_iter_outer=args.max_iter_outer,
                    alpha=alpha,
                    lambda_hsic=args.lambda_hsic,
                )
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "dataset": dataset_name,
                            "alpha": alpha,
                            "seed": seed,
                            "homogeneity": row["homogeneity"],
                            "nmi": row["nmi"],
                            "kmeans": row["kmeans"],
                            "silhouette": row["silhouette"],
                            "cka_score": row["cka_score"],
                            "runtime_s": row["fit_transform_runtime_s"],
                        }
                    ),
                    flush=True,
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
        "objective": "sdr",
        "alphas": args.alphas,
        "seeds": args.seeds,
        "output_sam": args.output_sam,
        "output_dim": args.output_dim,
        "lambda_hsic": args.lambda_hsic,
        "max_iter": args.max_iter,
        "max_iter_outer": args.max_iter_outer,
        "device": args.device,
        "hsic_kwargs": HSIC_KWARGS,
        "fgw_loss_kwargs": FGW_LOSS_KWARGS,
        "dataset_affinity_kwargs": DATASET_AFFINITY_KWARGS,
        "base_affinity_kwargs": {k: str(v) for k, v in BASE_AFFINITY_KWARGS.items()},
    }
    config_path.write_text(json.dumps(config, indent=2))

    print(f"Saved raw run metrics to {raw_path}")
    print(f"Saved summary metrics to {summary_path}")
    print(f"Saved run configuration to {config_path}")


if __name__ == "__main__":
    main()
