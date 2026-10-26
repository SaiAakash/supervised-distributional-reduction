import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gpytorch
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from benchmarks.benchmarks_dkl.gp_benchmarks import (
    SUPPORTED_DATASETS,
    _ExactGPModel,
    _MultitaskExactGPModel,
    _gp_split_metrics,
    _load_dataset_xy,
    _normalize_with_train_stats,
    _select_dataset_path,
    _set_seed,
    _train_exact_gp,
)
from src.affinities import NormalizedGaussianAndStudentAffinity, SymmetricEntropicAffinity
from src.sdr import SDR


DEFAULT_BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _summarize_metrics_across_seeds(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    metric_columns = [
        column
        for column in metrics_df.columns
        if column not in {"dataset", "seed", "model", "beta"}
        and pd.api.types.is_numeric_dtype(metrics_df[column])
    ]
    if not metric_columns:
        return pd.DataFrame()

    summary_df = (
        metrics_df.groupby(["dataset", "model", "beta"], as_index=False)[metric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary_df.columns = [
        (
            column
            if isinstance(column, str)
            else column[0] if column[1] == "" else f"{column[0]}_{column[1]}"
        )
        for column in summary_df.columns.to_flat_index()
    ]
    summary_df.insert(
        3,
        "num_seeds",
        metrics_df.groupby(["dataset", "model", "beta"]).size().to_numpy(),
    )
    return summary_df.sort_values(["dataset", "model", "beta"]).reset_index(drop=True)


def _summarize_runtimes_across_seeds(runtime_df: pd.DataFrame) -> pd.DataFrame:
    if runtime_df.empty:
        return pd.DataFrame()

    runtime_columns = [
        column
        for column in runtime_df.columns
        if column not in {"dataset", "seed", "model", "beta"}
        and pd.api.types.is_numeric_dtype(runtime_df[column])
    ]
    if not runtime_columns:
        return pd.DataFrame()

    summary_df = (
        runtime_df.groupby(["dataset", "model", "beta"], as_index=False)[runtime_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary_df.columns = [
        (
            column
            if isinstance(column, str)
            else column[0] if column[1] == "" else f"{column[0]}_{column[1]}"
        )
        for column in summary_df.columns.to_flat_index()
    ]
    summary_df.insert(
        3,
        "num_seeds",
        runtime_df.groupby(["dataset", "model", "beta"]).size().to_numpy(),
    )
    return summary_df.sort_values(["dataset", "model", "beta"]).reset_index(drop=True)


def benchmark_sdrgp_beta_sweep_dataset(
    dataset_name: str,
    seeds: list[int] | tuple[int, ...],
    beta_values: list[float] | tuple[float, ...] = DEFAULT_BETAS,
    data_path: str | Path | None = None,
    boston_path: str | Path | None = None,
    concrete_path: str | Path | None = None,
    energy_path: str | Path | None = None,
    test_size: float = 0.2,
    sdrgp_iters: int = 20,
    sdrgp_lr: float = 0.1,
    sdr_output_dim: int = 3,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.double,
    sdr_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_name = dataset_name.lower()
    selected_data_path = _select_dataset_path(
        dataset_name,
        data_path=data_path,
        boston_path=boston_path,
        concrete_path=concrete_path,
        energy_path=energy_path,
    )
    x_np, y_np = _load_dataset_xy(dataset_name, selected_data_path)
    device = torch.device(device)
    is_multitask = y_np.ndim > 1
    num_tasks = y_np.shape[-1] if is_multitask else 1

    base_sdr_kwargs: dict[str, Any] = {
        "affinity_data": SymmetricEntropicAffinity(perp=20, lr=0.01),
        "affinity_embedding": NormalizedGaussianAndStudentAffinity(student=True),
        "hsic_kernel_z": "rq",
        "hsic_kernel_y": "rbf",
        "fgw_loss": "square_loss",
        "loss_fun": "kl_loss",
        "alpha": 0.2,
        "output_dim": sdr_output_dim,
        "max_iter_outer": 20,
        "tol": 1e-5,
        "lambda_hsic": 1000.0,
        "device": device,
        "dtype": dtype,
        "verbose": False,
    }
    if sdr_overrides:
        base_sdr_kwargs.update(sdr_overrides)

    per_seed_results: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    normalized_betas = [float(beta) for beta in beta_values]

    for seed in seeds:
        _set_seed(int(seed))
        x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(
            x_np,
            y_np,
            test_size=test_size,
            random_state=int(seed),
        )
        train_x = torch.tensor(x_train_np, dtype=dtype, device=device)
        test_x = torch.tensor(x_test_np, dtype=dtype, device=device)
        train_y = torch.tensor(y_train_np, dtype=dtype, device=device)
        test_y = torch.tensor(y_test_np, dtype=dtype, device=device)
        if train_y.ndim == 1:
            train_y = train_y.unsqueeze(-1)
            test_y = test_y.unsqueeze(-1)

        train_x, test_x = _normalize_with_train_stats(train_x, test_x)
        train_y, test_y = _normalize_with_train_stats(train_y, test_y)

        seed_runs: list[dict[str, Any]] = []
        for beta in normalized_betas:
            sdr_kwargs = dict(base_sdr_kwargs)
            sdr_kwargs["output_sam"] = train_x.shape[-2]
            sdr_kwargs["seed"] = int(seed)
            sdr_kwargs["beta"] = beta
            dr = SDR(**sdr_kwargs)

            t0 = time.perf_counter()
            train_z, _ = dr.fit_transform(train_x, train_y)
            test_z = dr.transform(test_x)
            sdr_fit_s = time.perf_counter() - t0

            train_z, test_z = _normalize_with_train_stats(train_z, test_z)

            if is_multitask:
                sdrgp_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=num_tasks
                ).to(device=device, dtype=dtype)
                sdrgp_model = _MultitaskExactGPModel(
                    train_x=train_z,
                    train_y=train_y,
                    likelihood=sdrgp_likelihood,
                    ard_num_dims=sdr_output_dim,
                    num_tasks=num_tasks,
                ).to(device=device, dtype=dtype)
            else:
                sdrgp_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
                    device=device, dtype=dtype
                )
                sdrgp_model = _ExactGPModel(
                    train_x=train_z,
                    train_y=train_y.squeeze(-1),
                    likelihood=sdrgp_likelihood,
                    ard_num_dims=sdr_output_dim,
                ).to(device=device, dtype=dtype)

            t0 = time.perf_counter()
            sdrgp_logs = _train_exact_gp(
                model=sdrgp_model,
                likelihood=sdrgp_likelihood,
                train_x=train_z,
                train_y=train_y,
                n_iter=sdrgp_iters,
                lr=sdrgp_lr,
            )
            sdrgp_train_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            sdrgp_train_metrics = _gp_split_metrics(sdrgp_model, sdrgp_likelihood, train_z, train_y)
            sdrgp_test_metrics = _gp_split_metrics(sdrgp_model, sdrgp_likelihood, test_z, test_y)
            sdrgp_eval_s = time.perf_counter() - t0

            run_result = {
                "dataset": dataset_name,
                "seed": int(seed),
                "beta": beta,
                "model": "sdrgp",
                "runtimes_s": {
                    "sdr_fit": sdr_fit_s,
                    "sdrgp_train": sdrgp_train_s,
                    "sdrgp_eval": sdrgp_eval_s,
                    "total": sdr_fit_s + sdrgp_train_s + sdrgp_eval_s,
                },
                "metrics": {
                    "train": sdrgp_train_metrics,
                    "test": sdrgp_test_metrics,
                },
                "logs": {
                    "sdrgp": sdrgp_logs,
                },
            }
            seed_runs.append(run_result)

            metric_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "model": "sdrgp",
                    "beta": beta,
                    "train_mse": sdrgp_train_metrics["mse"],
                    "test_mse": sdrgp_test_metrics["mse"],
                    "train_mean_log_likelihood": sdrgp_train_metrics["mean_log_likelihood"],
                    "test_mean_log_likelihood": sdrgp_test_metrics["mean_log_likelihood"],
                }
            )
            runtime_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "model": "sdrgp",
                    "beta": beta,
                    **run_result["runtimes_s"],
                }
            )

        per_seed_results.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "beta_runs": seed_runs,
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    runtime_df = pd.DataFrame(runtime_rows)
    return {
        "dataset": dataset_name,
        "per_seed_results": per_seed_results,
        "metrics_df": metrics_df,
        "metrics_summary_df": _summarize_metrics_across_seeds(metrics_df),
        "runtime_df": runtime_df,
        "runtime_summary_df": _summarize_runtimes_across_seeds(runtime_df),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SDR-GP across different SDR beta values."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list(SUPPORTED_DATASETS),
        choices=SUPPORTED_DATASETS,
        help="Dataset names to benchmark.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="List of random seeds.",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=list(DEFAULT_BETAS),
        help="SDR beta values to sweep.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional dataset file path. Only valid when a single dataset is selected.",
    )
    parser.add_argument(
        "--boston-path",
        type=str,
        default=None,
        help="Optional path to a local Boston dataset file.",
    )
    parser.add_argument(
        "--concrete-path",
        type=str,
        default=None,
        help="Optional path to a local Concrete dataset file.",
    )
    parser.add_argument(
        "--energy-path",
        type=str,
        default=None,
        help="Optional path to a local Energy dataset file.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--sdrgp-iters", type=int, default=20)
    parser.add_argument("--sdrgp-lr", type=float, default=0.1)
    parser.add_argument("--sdr-output-dim", type=int, default=3)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_outputs/sdr_beta_benchmarks",
        help="Directory where result files are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_path is not None and len(args.datasets) != 1:
        raise ValueError("--data-path can only be used with a single dataset in --datasets.")

    dataset_paths = {
        "boston": args.boston_path,
        "concrete": args.concrete_path,
        "energy": args.energy_path,
    }

    config = {
        "datasets": args.datasets,
        "seeds": args.seeds,
        "betas": args.betas,
        "test_size": args.test_size,
        "sdrgp_iters": args.sdrgp_iters,
        "sdrgp_lr": args.sdrgp_lr,
        "sdr_output_dim": args.sdr_output_dim,
        "device": args.device,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    for dataset in args.datasets:
        data_path = dataset_paths[dataset]
        if data_path is None and len(args.datasets) == 1:
            data_path = args.data_path
        result = benchmark_sdrgp_beta_sweep_dataset(
            dataset_name=dataset,
            seeds=args.seeds,
            beta_values=args.betas,
            data_path=data_path,
            boston_path=args.boston_path,
            concrete_path=args.concrete_path,
            energy_path=args.energy_path,
            test_size=args.test_size,
            sdrgp_iters=args.sdrgp_iters,
            sdrgp_lr=args.sdrgp_lr,
            sdr_output_dim=args.sdr_output_dim,
            device=args.device,
        )

        prefix = dataset.lower().replace(" ", "_")
        metrics_path = output_dir / f"{prefix}_metrics.csv"
        metrics_summary_path = output_dir / f"{prefix}_metrics_summary.csv"
        runtime_path = output_dir / f"{prefix}_runtimes.csv"
        runtime_summary_path = output_dir / f"{prefix}_runtime_summary.csv"
        full_json_path = output_dir / f"{prefix}_full_results.json"

        result["metrics_df"].to_csv(metrics_path, index=False)
        result["metrics_summary_df"].to_csv(metrics_summary_path, index=False)
        result["runtime_df"].to_csv(runtime_path, index=False)
        result["runtime_summary_df"].to_csv(runtime_summary_path, index=False)

        serializable_result = {
            "dataset": result["dataset"],
            "per_seed_results": result["per_seed_results"],
            "metrics": result["metrics_df"].to_dict(orient="records"),
            "metrics_summary": result["metrics_summary_df"].to_dict(orient="records"),
            "runtimes": result["runtime_df"].to_dict(orient="records"),
            "runtime_summary": result["runtime_summary_df"].to_dict(orient="records"),
        }
        full_json_path.write_text(json.dumps(serializable_result, indent=2))

        print(f"[{dataset}] Wrote: {metrics_path}")
        print(f"[{dataset}] Wrote: {metrics_summary_path}")
        print(f"[{dataset}] Wrote: {runtime_path}")
        print(f"[{dataset}] Wrote: {runtime_summary_path}")
        print(f"[{dataset}] Wrote: {full_json_path}")
        print(f"\n[{dataset}] Metrics preview:")
        print(result["metrics_df"].to_string(index=False))


if __name__ == "__main__":
    main()
