import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import gpytorch
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmarks_dkl.gp_benchmarks import (
    _DKLModel,
    SUPPORTED_DATASETS,
    _FeatureExtractor,
    _MultitaskDKLModel,
    _flatten_targets,
    _gp_split_metrics,
    _load_dataset_xy,
    _normalize_with_train_stats,
    _resolve_dataset_path,
    _select_dataset_path,
    _set_seed,
    _split_summary_by_model,
    _summarize_metrics_across_seeds,
    _train_exact_gp,
)


class _FeatureExtractorRegressor(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        output_dim: int,
        feature_extractor: _FeatureExtractor | None = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor or _FeatureExtractor(input_dim, embedding_dim)
        self.regression_head = torch.nn.Linear(embedding_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regression_head(self.feature_extractor(x))


def _pretrain_feature_extractor(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    embedding_dim: int,
    n_iter: int,
    lr: float,
) -> tuple[_FeatureExtractor, list[dict[str, float]]]:
    output_dim = train_y.shape[-1]
    regressor = _FeatureExtractorRegressor(
        input_dim=train_x.shape[-1],
        embedding_dim=embedding_dim,
        output_dim=output_dim,
    ).to(device=train_x.device, dtype=train_x.dtype)
    optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)

    logs: list[dict[str, float]] = []
    for i in range(n_iter):
        optimizer.zero_grad()
        pred = regressor(train_x)
        mse = F.mse_loss(pred, train_y)
        mse.backward()
        optimizer.step()
        logs.append({"iter": float(i + 1), "mse": float(mse.item())})

    return regressor.feature_extractor, logs


def _build_pretrained_dkl_parameter_groups(
    model: gpytorch.models.ExactGP,
    freeze_feature_extractor: bool,
) -> list[dict[str, Any]]:
    if freeze_feature_extractor:
        for param in model.feature_extractor.parameters():
            param.requires_grad_(False)
    else:
        for param in model.feature_extractor.parameters():
            param.requires_grad_(True)

    parameter_groups: list[dict[str, Any]] = []
    if not freeze_feature_extractor:
        parameter_groups.append({"params": model.feature_extractor.parameters()})
    parameter_groups.extend(
        [
            {"params": model.covar_module.parameters()},
            {"params": model.mean_module.parameters()},
            {"params": model.likelihood.parameters()},
        ]
    )
    return parameter_groups


def benchmark_pretrained_dkl_dataset(
    dataset_name: str,
    seeds: list[int] | tuple[int, ...],
    data_path: str | Path | None = None,
    boston_path: str | Path | None = None,
    concrete_path: str | Path | None = None,
    energy_path: str | Path | None = None,
    test_size: float = 0.2,
    pretrain_iters: int = 100,
    pretrain_lr: float = 1e-3,
    gp_iters: int = 20,
    gp_lr: float = 0.01,
    embedding_dim: int = 2,
    freeze_feature_extractor: bool = False,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.double,
) -> dict[str, Any]:
    dataset_name = dataset_name.lower()
    selected_data_path = _select_dataset_path(
        dataset_name,
        data_path=data_path,
        boston_path=boston_path,
        concrete_path=concrete_path,
        energy_path=energy_path,
    )
    X, y = _load_dataset_xy(dataset_name, selected_data_path)
    device = torch.device(device)
    is_multitask = y.ndim > 1
    num_tasks = y.shape[-1] if is_multitask else 1

    per_seed_results: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for seed in seeds:
        _set_seed(int(seed))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(seed)
        )
        train_x = torch.tensor(X_train, dtype=dtype, device=device)
        test_x = torch.tensor(X_test, dtype=dtype, device=device)
        train_y = torch.tensor(y_train, dtype=dtype, device=device)
        test_y = torch.tensor(y_test, dtype=dtype, device=device)
        if train_y.ndim == 1:
            train_y = train_y.unsqueeze(-1)
            test_y = test_y.unsqueeze(-1)

        train_x, test_x = _normalize_with_train_stats(train_x, test_x)
        train_y, test_y = _normalize_with_train_stats(train_y, test_y)

        seed_runtimes: dict[str, float] = {}

        t0 = time.perf_counter()
        pretrained_feature_extractor, pretrain_logs = _pretrain_feature_extractor(
            train_x=train_x,
            train_y=train_y,
            embedding_dim=embedding_dim,
            n_iter=pretrain_iters,
            lr=pretrain_lr,
        )
        seed_runtimes["pretrained_dkl_pretrain"] = time.perf_counter() - t0

        if is_multitask:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(
                device=device, dtype=dtype
            )
            model = _MultitaskDKLModel(
                train_x=train_x,
                train_y=train_y,
                likelihood=likelihood,
                input_dim=train_x.shape[-1],
                embedding_dim=embedding_dim,
                num_tasks=num_tasks,
            ).to(device=device, dtype=dtype)
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
            model = _DKLModel(
                train_x=train_x,
                train_y=_flatten_targets(train_y),
                likelihood=likelihood,
                input_dim=train_x.shape[-1],
                embedding_dim=embedding_dim,
            ).to(device=device, dtype=dtype)
        model.feature_extractor = pretrained_feature_extractor

        parameter_groups = _build_pretrained_dkl_parameter_groups(
            model=model,
            freeze_feature_extractor=freeze_feature_extractor,
        )

        t0 = time.perf_counter()
        gp_logs = _train_exact_gp(
            model=model,
            likelihood=likelihood,
            train_x=train_x,
            train_y=train_y,
            n_iter=gp_iters,
            lr=gp_lr,
            parameter_groups=parameter_groups,
        )
        seed_runtimes["pretrained_dkl_train"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        train_metrics = _gp_split_metrics(model, likelihood, train_x, train_y)
        test_metrics = _gp_split_metrics(model, likelihood, test_x, test_y)
        seed_runtimes["pretrained_dkl_eval"] = time.perf_counter() - t0
        seed_runtimes["total"] = float(sum(seed_runtimes.values()))

        seed_result = {
            "dataset": dataset_name,
            "seed": int(seed),
            "methods": ["pretrained_dkl"],
            "runtimes_s": seed_runtimes,
            "metrics": {"pretrained_dkl": {"train": train_metrics, "test": test_metrics}},
            "logs": {"pretrained_dkl": {"pretrain": pretrain_logs, "gp": gp_logs}},
        }
        per_seed_results.append(seed_result)
        metric_rows.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "model": "pretrained_dkl",
                "train_mse": train_metrics["mse"],
                "test_mse": test_metrics["mse"],
                "train_mean_log_likelihood": train_metrics["mean_log_likelihood"],
                "test_mean_log_likelihood": test_metrics["mean_log_likelihood"],
            }
        )
        runtime_rows.append({"dataset": dataset_name, "seed": int(seed), **seed_runtimes})

    metrics_df = pd.DataFrame(metric_rows)
    metrics_summary_df = _summarize_metrics_across_seeds(metrics_df)
    return {
        "dataset": dataset_name,
        "per_seed_results": per_seed_results,
        "metrics_df": metrics_df,
        "metrics_summary_df": metrics_summary_df,
        "metrics_summary_by_model": _split_summary_by_model(metrics_summary_df),
        "runtime_df": pd.DataFrame(runtime_rows),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pre-trained DKL regression benchmarks using the gp_benchmarks feature extractor."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["boston"],
        choices=SUPPORTED_DATASETS,
        help="Dataset names to benchmark.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="List of random seeds.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional dataset file path. Only valid when a single dataset is selected.",
    )
    parser.add_argument("--boston-path", type=str, default=None)
    parser.add_argument("--concrete-path", type=str, default=None)
    parser.add_argument("--energy-path", type=str, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--pretrain-iters", type=int, default=75)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--gp-iters", type=int, default=20)
    parser.add_argument("--gp-lr", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=2)
    parser.add_argument(
        "--freeze-feature-extractor",
        action="store_true",
        help="Freeze the pre-trained feature extractor during GP fitting.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_outputs/pretrained_dkl_regression_benchmarks",
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

    for dataset in args.datasets:
        data_path = dataset_paths[dataset]
        if data_path is None and len(args.datasets) == 1:
            data_path = args.data_path
        result = benchmark_pretrained_dkl_dataset(
            dataset_name=dataset,
            seeds=args.seeds,
            data_path=data_path,
            test_size=args.test_size,
            pretrain_iters=args.pretrain_iters,
            pretrain_lr=args.pretrain_lr,
            gp_iters=args.gp_iters,
            gp_lr=args.gp_lr,
            embedding_dim=args.embedding_dim,
            freeze_feature_extractor=args.freeze_feature_extractor,
            device=args.device,
        )

        prefix = dataset.lower().replace(" ", "_")
        metrics_path = output_dir / f"{prefix}_metrics.csv"
        runtime_path = output_dir / f"{prefix}_runtimes.csv"
        full_json_path = output_dir / f"{prefix}_full_results.json"

        result["metrics_df"].to_csv(metrics_path, index=False)
        summary_paths: dict[str, Path] = {}
        for model_name, summary_df in result["metrics_summary_by_model"].items():
            summary_path = output_dir / f"{prefix}_{model_name}_metrics_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            summary_paths[model_name] = summary_path
        result["runtime_df"].to_csv(runtime_path, index=False)

        serializable_result = {
            "dataset": result["dataset"],
            "methods": ["pretrained_dkl"],
            "per_seed_results": result["per_seed_results"],
            "metrics": result["metrics_df"].to_dict(orient="records"),
            "metrics_summary_by_model": {
                model_name: summary_df.to_dict(orient="records")
                for model_name, summary_df in result["metrics_summary_by_model"].items()
            },
            "runtimes": result["runtime_df"].to_dict(orient="records"),
        }
        full_json_path.write_text(json.dumps(serializable_result, indent=2))

        resolved_path = _resolve_dataset_path(dataset, data_path)
        print(f"[{dataset}] Dataset: {resolved_path}")
        print(f"[{dataset}] Method: pretrained_dkl")
        print(f"[{dataset}] Wrote: {metrics_path}")
        for model_name, summary_path in summary_paths.items():
            print(f"[{dataset}] Wrote: {summary_path}")
        print(f"[{dataset}] Wrote: {runtime_path}")
        print(f"[{dataset}] Wrote: {full_json_path}")
        print(f"\n[{dataset}] Metrics preview:")
        print(result["metrics_df"].to_string(index=False))


if __name__ == "__main__":
    main()
