import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_friedman1, make_s_curve, make_swiss_roll
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from src.affinities import NormalizedGaussianAndStudentAffinity, SymmetricEntropicAffinity
from src.clust_dr import DistR
from src.kernel_supervised_pca import KernelSupervisedPCA
from src.sdr import SDR


DEFAULT_DATASETS = ("swiss_roll", "s_curve", "friedman1")
DEFAULT_METHODS = ("sdr", "supervised_umap", "kernel_supervised_pca", "distr")
DEFAULT_SEEDS = tuple(range(5))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _standardize_train_test(
    train_array: np.ndarray,
    test_array: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_array)
    test_scaled = scaler.transform(test_array)
    return (
        torch.tensor(train_scaled, dtype=torch.double),
        torch.tensor(test_scaled, dtype=torch.double),
    )


def _make_dataset(
    dataset_name: str,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if dataset_name == "swiss_roll":
        x, y = make_swiss_roll(
            n_samples=n_samples,
            noise=0.0,
            random_state=seed,
            hole=True,
        )
        return x.astype(np.float64), y.astype(np.float64)
    if dataset_name == "s_curve":
        x, y = make_s_curve(
            n_samples=n_samples,
            noise=0.0,
            random_state=seed,
        )
        return x.astype(np.float64), y.astype(np.float64)
    if dataset_name == "friedman1":
        x, y = make_friedman1(
            n_samples=n_samples,
            n_features=10,
            noise=0.0,
            random_state=seed,
        )
        return x.astype(np.float64), y.astype(np.float64)
    raise ValueError(f"Unsupported dataset {dataset_name!r}")


def _make_sdr(
    seed: int,
    output_sam: int,
    output_dim: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
) -> SDR:
    return SDR(
        affinity_data=SymmetricEntropicAffinity(perp=50, lr=0.01, verbose=False),
        affinity_embedding=NormalizedGaussianAndStudentAffinity(student=True),
        hsic_kernel_z="rq",
        hsic_kernel_y="rbf",
        fgw_loss="square_loss",
        loss_fun="kl_loss",
        alpha=0.2,
        output_sam=output_sam,
        output_dim=output_dim,
        optimizer="Adam",
        init="normal",
        tol=1e-5,
        verbose=False,
        seed=seed,
        device=device,
        dtype=torch.double,
        max_iter=max_iter,
        max_iter_outer=max_iter_outer,
        init_T="spectral",
        lambda_hsic=1000.0,
    )


def _make_distr(
    seed: int,
    output_sam: int,
    output_dim: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
) -> DistR:
    return DistR(
        affinity_data=SymmetricEntropicAffinity(perp=50, lr=0.01, verbose=False),
        affinity_embedding=NormalizedGaussianAndStudentAffinity(student=True),
        loss_fun="kl_loss",
        alpha=1.0,
        output_sam=output_sam,
        output_dim=output_dim,
        optimizer="Adam",
        init="normal",
        tol=1e-5,
        verbose=False,
        seed=seed,
        device=device,
        dtype=torch.double,
        max_iter=max_iter,
        max_iter_outer=max_iter_outer,
        init_T="spectral",
    )


def _evaluate_regressors(
    train_z: np.ndarray,
    test_z: Optional[np.ndarray],
    train_y: np.ndarray,
    test_y: Optional[np.ndarray],
    knn_neighbors: int,
    krr_alpha: float,
) -> list[dict[str, float | str]]:
    scaler = StandardScaler()
    train_z_scaled = scaler.fit_transform(train_z)
    test_z_scaled = None if test_z is None else scaler.transform(test_z)

    models = {
        "knn": KNeighborsRegressor(n_neighbors=knn_neighbors),
        "krr": KernelRidge(kernel="rbf", alpha=krr_alpha),
    }

    rows: list[dict[str, float | str]] = []
    for model_name, model in models.items():
        model.fit(train_z_scaled, train_y)
        rows.append(
            {
                "regressor": model_name,
                "train_score": float(model.score(train_z_scaled, train_y)),
                "test_score": (
                    float(model.score(test_z_scaled, test_y))
                    if test_z_scaled is not None and test_y is not None
                    else np.nan
                ),
            }
        )
    return rows


def _compute_embedding(
    method_name: str,
    seed: int,
    train_x: torch.Tensor,
    test_x: torch.Tensor,
    train_y: torch.Tensor,
    output_dim: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
) -> tuple[np.ndarray, Optional[np.ndarray], float]:
    start = time.perf_counter()

    if method_name == "sdr":
        method = _make_sdr(
            seed=seed,
            output_sam=train_x.shape[0],
            output_dim=output_dim,
            device=device,
            max_iter=max_iter,
            max_iter_outer=max_iter_outer,
        )
        method.fit_transform(train_x.to(device), train_y.to(device))
        train_z = method.transform(train_x.to(device)).detach().cpu().numpy()
        test_z = method.transform(test_x.to(device)).detach().cpu().numpy()
    elif method_name == "distr":
        method = _make_distr(
            seed=seed,
            output_sam=train_x.shape[0],
            output_dim=output_dim,
            device=device,
            max_iter=max_iter,
            max_iter_outer=max_iter_outer,
        )
        train_z = method.fit_transform(train_x.to(device)).detach().cpu().numpy()
        test_z = None
    elif method_name == "supervised_umap":
        os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
        import umap

        reducer = umap.UMAP(
            n_neighbors=50,
            min_dist=0.1,
            n_components=output_dim,
            random_state=seed,
            target_metric="l2",
        )
        train_x_np = train_x.detach().cpu().numpy()
        test_x_np = test_x.detach().cpu().numpy()
        train_y_np = train_y.detach().cpu().numpy().ravel()
        train_z = reducer.fit_transform(train_x_np, y=train_y_np)
        test_z = reducer.transform(test_x_np)
    elif method_name == "kernel_supervised_pca":
        train_x_kspca = train_x.to(dtype=torch.float32)
        test_x_kspca = test_x.to(dtype=torch.float32)
        train_y_kspca = train_y.to(dtype=torch.float32)
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            method = KernelSupervisedPCA(
                n_components=output_dim,
                kernel="rbf",
                target_kernel="rbf",
                reg=1e-6,
                device=device,
            )
            train_z = method.fit_transform(train_x_kspca, train_y_kspca).detach().cpu().numpy()
            test_z = method.transform(test_x_kspca).detach().cpu().numpy()
        finally:
            torch.set_default_dtype(default_dtype)
    else:
        raise ValueError(f"Unsupported method {method_name!r}")

    runtime_s = time.perf_counter() - start
    return train_z, test_z, runtime_s


def _run_single_benchmark(
    method_name: str,
    dataset_name: str,
    seed: int,
    n_samples: int,
    test_size: float,
    output_dim: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
    knn_neighbors: int,
    krr_alpha: float,
) -> list[dict[str, float | int | str]]:
    _set_seed(seed)
    x, y = _make_dataset(dataset_name=dataset_name, n_samples=n_samples, seed=seed)
    train_x_np, test_x_np, train_y_np, test_y_np = train_test_split(
        x,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=seed,
    )

    train_x, test_x = _standardize_train_test(train_x_np, test_x_np)
    train_y, test_y = _standardize_train_test(
        train_y_np.reshape(-1, 1),
        test_y_np.reshape(-1, 1),
    )

    train_z, test_z, runtime_s = _compute_embedding(
        method_name=method_name,
        seed=seed,
        train_x=train_x,
        test_x=test_x,
        train_y=train_y,
        output_dim=output_dim,
        device=device,
        max_iter=max_iter,
        max_iter_outer=max_iter_outer,
    )
    train_y_np = train_y.detach().cpu().numpy().ravel()
    test_y_np = test_y.detach().cpu().numpy().ravel()

    rows: list[dict[str, float | int | str]] = []
    for score_row in _evaluate_regressors(
        train_z=train_z,
        test_z=test_z,
        train_y=train_y_np,
        test_y=test_y_np if test_z is not None else None,
        knn_neighbors=knn_neighbors,
        krr_alpha=krr_alpha,
    ):
        rows.append(
            {
                "dataset": dataset_name,
                "method": method_name,
                "seed": seed,
                "regressor": score_row["regressor"],
                "n_samples": n_samples,
                "n_train": int(train_x.shape[0]),
                "n_test": int(test_x.shape[0]),
                "input_dim": int(train_x.shape[1]),
                "output_dim": output_dim,
                "train_score": score_row["train_score"],
                "test_score": score_row["test_score"],
                "fit_transform_runtime_s": runtime_s,
            }
        )
    return rows


def _aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby(
            ["dataset", "method", "regressor", "n_samples", "output_dim"],
            as_index=False,
        )
        .agg(
            train_score_mean=("train_score", "mean"),
            train_score_std=("train_score", "std"),
            test_score_mean=("test_score", "mean"),
            test_score_std=("test_score", "std"),
            runtime_mean_s=("fit_transform_runtime_s", "mean"),
            runtime_std_s=("fit_transform_runtime_s", "std"),
        )
        .sort_values(["dataset", "method", "regressor"])
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pure-DR toy regression benchmarks and evaluate KNN/KRR downstream "
            "performance for multiple embedding methods."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--max-iter-outer", type=int, default=20)
    parser.add_argument("--knn-neighbors", type=int, default=10)
    parser.add_argument("--krr-alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "benchmark_outputs" / "pure_dr_toy_benchmarks"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    unknown_datasets = set(args.datasets) - set(DEFAULT_DATASETS)
    if unknown_datasets:
        raise ValueError(f"Unsupported datasets requested: {sorted(unknown_datasets)}")
    unknown_methods = set(args.methods) - set(DEFAULT_METHODS)
    if unknown_methods:
        raise ValueError(f"Unsupported methods requested: {sorted(unknown_methods)}")

    torch.set_default_dtype(torch.double)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for dataset_name in args.datasets:
        for method_name in args.methods:
            for seed in args.seeds:
                run_rows = _run_single_benchmark(
                    method_name=method_name,
                    dataset_name=dataset_name,
                    seed=seed,
                    n_samples=args.n_samples,
                    test_size=args.test_size,
                    output_dim=args.output_dim,
                    device=args.device,
                    max_iter=args.max_iter,
                    max_iter_outer=args.max_iter_outer,
                    knn_neighbors=args.knn_neighbors,
                    krr_alpha=args.krr_alpha,
                )
                rows.extend(run_rows)
                for row in run_rows:
                    print(json.dumps(row))

    results_df = pd.DataFrame(rows)
    summary_df = _aggregate_results(results_df)

    raw_path = results_dir / "run_metrics.csv"
    summary_path = results_dir / "summary.csv"
    config_path = results_dir / "config.json"

    results_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "datasets": args.datasets,
                "methods": args.methods,
                "seeds": args.seeds,
                "n_samples": args.n_samples,
                "test_size": args.test_size,
                "output_dim": args.output_dim,
                "max_iter": args.max_iter,
                "max_iter_outer": args.max_iter_outer,
                "knn_neighbors": args.knn_neighbors,
                "krr_alpha": args.krr_alpha,
                "device": args.device,
            },
            indent=2,
        )
    )

    print(f"Saved raw run metrics to {raw_path}")
    print(f"Saved summary metrics to {summary_path}")
    print(f"Saved run configuration to {config_path}")


if __name__ == "__main__":
    main()
