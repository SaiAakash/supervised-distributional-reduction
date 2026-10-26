import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.affinities import NormalizedGaussianAndStudentAffinity, SymmetricEntropicAffinity
from src.hsic_loss import hsic
from src.scores import scores_clustdr
from src.sdr import SDR

from benchmarks.benchmarks_clustdr.classification_joint_benchmarks import _preprocess_features


DEFAULT_DATASETS = ("coil20", "snareseq")
DEFAULT_OBJECTIVES = ("sdr", "fgw")
DEFAULT_SEEDS = tuple(range(5))

BASE_SDR_KWARGS = {
    "affinity_embedding": NormalizedGaussianAndStudentAffinity(student=True),
    "hsic_kernel_z": "rq",
    "hsic_kernel_y": "delta",
    "fgw_loss": "ce_loss",
    "loss_fun": "kl_loss",
    "optimizer": "Adam",
    "init": "normal",
    "tol": 1e-5,
    "verbose": False,
    "init_T": "spectral",
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


DATASET_LOADERS = {
    "coil20": _load_coil20,
    "snareseq": _load_snareseq,
}


def _make_sdr(
    dataset_name: str,
    objective: str,
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
        **deepcopy(BASE_SDR_KWARGS),
        affinity_data=SymmetricEntropicAffinity(**affinity_kwargs, verbose=False),
        objective=objective,
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


def _final_loss_terms(method: SDR) -> dict[str, float]:
    terms = method.last_loss_terms or {}
    return {
        "final_objective": float(terms.get("objective", np.nan)),
        "final_structural_loss": float(terms.get("structural_loss", np.nan)),
        "final_supervision_loss": float(terms.get("supervision_loss", np.nan)),
    }


def _compute_cka_score(method: SDR, y: torch.Tensor, z: torch.Tensor) -> float:
    if method.use_rff:
        hsic_kwargs = {"sigma": 1.0}
    else:
        hsic_kwargs = {
            "sigma": None,
            "alpha": 1.0,
        }

    t = method.T.detach() if method.output_sam < y.shape[0] else None
    cka_score = hsic(
        z.detach(),
        y.detach(),
        T=t,
        kernel_z=method.hsic_kernel_z,
        kernel_y=method.hsic_kernel_y,
        device=method.device,
        dtype=method.dtype,
        **hsic_kwargs,
    )
    return float(cka_score.item())


def _run_single(
    dataset_name: str,
    objective: str,
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
        objective=objective,
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
        "objective": objective,
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
        "inner_steps": len(method.inner_loss_history),
        "outer_steps": len(method.outer_loss_history),
    }


def _aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby(
            ["dataset", "objective", "output_sam", "output_dim", "alpha", "lambda_hsic"],
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
            cka_score_mean=("cka_score", "mean"),
            cka_score_std=("cka_score", "std"),
            final_objective_mean=("final_objective", "mean"),
            final_objective_std=("final_objective", "std"),
            final_structural_loss_mean=("final_structural_loss", "mean"),
            final_structural_loss_std=("final_structural_loss", "std"),
            final_supervision_loss_mean=("final_supervision_loss", "mean"),
            final_supervision_loss_std=("final_supervision_loss", "std"),
            runtime_mean_s=("fit_transform_runtime_s", "mean"),
            runtime_std_s=("fit_transform_runtime_s", "std"),
            inner_steps_mean=("inner_steps", "mean"),
            outer_steps_mean=("outer_steps", "mean"),
            active_prototypes_mean=("active_prototypes", "mean"),
            active_prototypes_std=("active_prototypes", "std"),
        )
        .sort_values(["dataset", "objective", "output_sam"])
    )


def _write_objective_specific_results(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    results_dir: Path,
) -> list[tuple[str, Path, Path]]:
    written_paths = []
    for objective in sorted(results_df["objective"].unique()):
        objective_results = results_df[results_df["objective"] == objective].copy()
        objective_summary = summary_df[summary_df["objective"] == objective].copy()
        raw_path = results_dir / f"run_metrics_{objective}.csv"
        summary_path = results_dir / f"summary_{objective}.csv"
        objective_results.to_csv(raw_path, index=False)
        objective_summary.to_csv(summary_path, index=False)
        written_paths.append((objective, raw_path, summary_path))
    return written_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SDR and FGW objectives on COIL-20 and SNAREseq."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--objectives", nargs="+", default=list(DEFAULT_OBJECTIVES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-sam", type=int, default=200)
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--lambda-hsic", type=float, default=1000.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--max-iter-outer", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "benchmark_outputs" / "sdr_objective_comparison"),
    )
    parser.add_argument("--coil20-path")
    parser.add_argument("--snareseq-path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    unknown_datasets = set(args.datasets) - set(DATASET_LOADERS)
    if unknown_datasets:
        raise ValueError(f"Unsupported datasets requested: {sorted(unknown_datasets)}")

    unknown_objectives = set(args.objectives) - {"sdr", "fgw"}
    if unknown_objectives:
        raise ValueError(f"Unsupported objectives requested: {sorted(unknown_objectives)}")

    dataset_path_map = {
        "coil20": args.coil20_path,
        "snareseq": args.snareseq_path,
    }

    torch.set_default_dtype(torch.double)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cached_datasets = {}
    for dataset_name in args.datasets:
        x, y = DATASET_LOADERS[dataset_name](dataset_path_map[dataset_name])
        x = _preprocess_features(x)
        cached_datasets[dataset_name] = (x.to(args.device), y.to(args.device))

    rows = []
    for dataset_name in args.datasets:
        x, y = cached_datasets[dataset_name]
        if args.output_sam > x.shape[0]:
            raise ValueError(
                f"output_sam={args.output_sam} exceeds sample count {x.shape[0]} for {dataset_name}."
            )

        for objective in args.objectives:
            for seed in args.seeds:
                row = _run_single(
                    dataset_name=dataset_name,
                    objective=objective,
                    x=x,
                    y=y,
                    seed=seed,
                    output_sam=args.output_sam,
                    output_dim=args.output_dim,
                    device=args.device,
                    max_iter=args.max_iter,
                    max_iter_outer=args.max_iter_outer,
                    alpha=args.alpha,
                    lambda_hsic=args.lambda_hsic,
                )
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "dataset": dataset_name,
                            "objective": objective,
                            "seed": seed,
                            "cka_score": row["cka_score"],
                            "homogeneity": row["homogeneity"],
                            "nmi": row["nmi"],
                            "silhouette": row["silhouette"],
                            "kmeans": row["kmeans"],
                            "final_objective": row["final_objective"],
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
    per_objective_paths = _write_objective_specific_results(results_df, summary_df, results_dir)

    config = {
        "datasets": args.datasets,
        "objectives": args.objectives,
        "seeds": args.seeds,
        "output_sam": args.output_sam,
        "output_dim": args.output_dim,
        "alpha": args.alpha,
        "lambda_hsic": args.lambda_hsic,
        "max_iter": args.max_iter,
        "max_iter_outer": args.max_iter_outer,
        "device": args.device,
    }
    config_path.write_text(json.dumps(config, indent=2))

    print(f"Saved raw run metrics to {raw_path}")
    print(f"Saved summary metrics to {summary_path}")
    for objective, objective_raw_path, objective_summary_path in per_objective_paths:
        print(f"Saved {objective} raw run metrics to {objective_raw_path}")
        print(f"Saved {objective} summary metrics to {objective_summary_path}")


if __name__ == "__main__":
    main()
