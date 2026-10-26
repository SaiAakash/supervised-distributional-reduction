import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmarks_dkl.gp_benchmarks import (
    _DKLModel,
    _ExactGPModel,
    _MultitaskDKLModel,
    _MultitaskExactGPModel,
    _RegressionDeepGPModel,
    _deep_gp_split_metrics,
    _fit_nca_embeddings,
    _gp_split_metrics,
    _load_dataset_xy,
    _mean_over_leading_dims,
    _multitask_deep_gp_split_metrics,
    _select_dataset_path,
    _set_seed,
    _train_deep_gp,
    _train_exact_gp,
    _train_multitask_deep_gp,
)
from benchmarks.benchmarks_dkl.metrics import calibration_curve, calibration_error
from src.affinities import NormalizedGaussianAndStudentAffinity, SymmetricEntropicAffinity
from src.kernel_supervised_pca import KernelSupervisedPCA
from src.sdr import SDR


DATASET_ORDER = ("boston", "energy", "concrete")
CALIBRATION_TITLE_FONT_SIZE = 20
CALIBRATION_AXIS_LABEL_FONT_SIZE = 15
CALIBRATION_TICK_FONT_SIZE = 15
CALIBRATION_LEGEND_FONT_SIZE = 20

TRADEOFF_TITLE_FONT_SIZE = 40
TRADEOFF_AXIS_LABEL_FONT_SIZE = 30
TRADEOFF_TICK_FONT_SIZE = 25
TRADEOFF_LEGEND_FONT_SIZE = 30

MODEL_ORDER = (
    "kernel_supervised_pca_gp",
    "nca_gp",
    "supervised_umap_gp",
    "deep_gp",
    "dkl",
    "sdrgp",
)
MODEL_LABELS = {
    "kernel_supervised_pca_gp": "KSPCA-GP",
    "nca_gp": "NCA-GP",
    "supervised_umap_gp": "UMAP-GP",
    "deep_gp": "Deep GP",
    "dkl": "DKL",
    "sdrgp": "SDR-GP",
}
DATASET_LABELS = {
    "boston": "Boston",
    "energy": "Energy",
    "concrete": "Concrete",
}
MODEL_MARKERS = {
    "kernel_supervised_pca_gp": "o",
    "nca_gp": "X",
    "supervised_umap_gp": "P",
    "deep_gp": "^",
    "dkl": "D",
    "sdrgp": "s",
}
MODEL_COLORS = {
    "kernel_supervised_pca_gp": "#1f77b4",
    "nca_gp": "#8c564b",
    "supervised_umap_gp": "#ff7f0e",
    "deep_gp": "#2ca02c",
    "dkl": "#9467bd",
    "sdrgp": "#d62728",
}

def _normalize_with_stats(
    train_t: torch.Tensor, test_t: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = train_t.mean(0, keepdim=True)
    std = train_t.std(0, keepdim=True).clamp_min(eps)
    return (train_t - mu) / std, (test_t - mu) / std, mu, std


def _fit_supervised_umap_embeddings(
    train_x: torch.Tensor,
    test_x: torch.Tensor,
    train_y: torch.Tensor,
    output_dim: int,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    target_metric: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=output_dim,
        random_state=seed,
        target_metric=target_metric,
    )
    train_y_np = train_y.detach().cpu().numpy()
    train_z_np = reducer.fit_transform(
        train_x.detach().cpu().numpy(),
        y=train_y_np if train_y_np.ndim > 1 else train_y_np.ravel(),
    )
    test_z_np = reducer.transform(test_x.detach().cpu().numpy())
    train_z = torch.as_tensor(train_z_np, device=train_x.device, dtype=train_x.dtype)
    test_z = torch.as_tensor(test_z_np, device=test_x.device, dtype=test_x.dtype)
    return train_z, test_z


def _fit_kernel_supervised_pca_embeddings(
    train_x: torch.Tensor,
    test_x: torch.Tensor,
    train_y: torch.Tensor,
    output_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    kernel: str,
    gamma: float | None,
    target_kernel: str,
    target_gamma: float | None,
    reg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_x_kspca = train_x.to(dtype=torch.float32)
    test_x_kspca = test_x.to(dtype=torch.float32)
    train_y_kspca = train_y.to(dtype=torch.float32)
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        reducer = KernelSupervisedPCA(
            n_components=output_dim,
            kernel=kernel,
            gamma=gamma,
            target_kernel=target_kernel,
            target_gamma=target_gamma,
            reg=reg,
            device=device,
        )
        train_z = reducer.fit_transform(train_x_kspca, train_y_kspca)
        test_z = reducer.transform(test_x_kspca)
    finally:
        torch.set_default_dtype(default_dtype)
    return train_z.to(device=device, dtype=dtype), test_z.to(device=device, dtype=dtype)


def _predict_exact_gp(
    model: torch.nn.Module,
    likelihood: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred = likelihood(model(x))
        return pred.mean, pred.variance.clamp_min(1e-8).sqrt()


def _predict_deep_gp(
    model: _RegressionDeepGPModel,
    x: torch.Tensor,
    num_likelihood_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    import gpytorch

    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(num_likelihood_samples):
        pred = model.likelihood(model(x))
        mean = _mean_over_leading_dims(pred.mean, target_ndim=1)
        std = _mean_over_leading_dims(pred.variance, target_ndim=1).clamp_min(1e-8).sqrt()
    return mean, std


def _predict_multitask_deep_gp(
    models: list[_RegressionDeepGPModel],
    x: torch.Tensor,
    num_likelihood_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    means: list[torch.Tensor] = []
    stds: list[torch.Tensor] = []
    for task_idx, model in enumerate(models):
        with torch.random.fork_rng():
            torch.manual_seed(task_idx)
            mean, std = _predict_deep_gp(model, x, num_likelihood_samples)
        means.append(mean)
        stds.append(std)
    return torch.stack(means, dim=-1), torch.stack(stds, dim=-1)


def _to_original_scale(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_std: torch.Tensor,
    target_mu: torch.Tensor,
    target_std: torch.Tensor,
    y_train: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_true_orig = y_true * target_std + target_mu
    y_pred_orig = y_pred * target_std + target_mu
    y_std_orig = y_std * target_std
    y_train_orig = y_train * target_std + target_mu
    return (
        y_true_orig.detach().cpu().numpy(),
        y_pred_orig.detach().cpu().numpy(),
        y_std_orig.detach().cpu().numpy(),
        y_train_orig.detach().cpu().numpy(),
    )


def _plot_mll_vs_mace_tradeoff(
    summary_df: pd.DataFrame,
    datasets: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(7.0 * len(datasets), 7.2),
        squeeze=False,
        sharey=True,
    )

    for col_idx, dataset_name in enumerate(datasets):
        ax = axes[0][col_idx]
        dataset_df = summary_df[summary_df["dataset"] == dataset_name].copy()
        dataset_df["model_order"] = dataset_df["model"].map(
            {model_name: idx for idx, model_name in enumerate(MODEL_ORDER)}
        )
        dataset_df = dataset_df.sort_values("model_order")

        for _, row in dataset_df.iterrows():
            model_name = str(row["model"])
            x_value = float(row["mean_log_likelihood_mean"])
            y_value = -float(row["mace_interval_mean"])
            ax.scatter(
                x_value,
                y_value,
                s=150,
                marker=MODEL_MARKERS[model_name],
                color=MODEL_COLORS[model_name],
                edgecolor="black",
                linewidth=0.9,
                alpha=0.95,
                zorder=3,
            )
            xerr = row.get("mean_log_likelihood_std")
            yerr = row.get("mace_interval_std")
            if pd.notna(xerr) or pd.notna(yerr):
                ax.errorbar(
                    x_value,
                    y_value,
                    xerr=float(xerr) if pd.notna(xerr) else None,
                    yerr=float(yerr) if pd.notna(yerr) else None,
                    color=MODEL_COLORS[model_name],
                    ecolor=MODEL_COLORS[model_name],
                    elinewidth=1.5,
                    capsize=4,
                    alpha=0.8,
                    zorder=2,
                )

        ax.set_title(
            DATASET_LABELS.get(dataset_name, dataset_name.title()),
            fontsize=TRADEOFF_TITLE_FONT_SIZE,
            pad=18,
        )
        ax.set_xlabel("MLL", fontsize=TRADEOFF_AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel("-MACE", fontsize=TRADEOFF_AXIS_LABEL_FONT_SIZE)
        ax.tick_params(axis="both", which="both", labelsize=TRADEOFF_TICK_FONT_SIZE)
        ax.grid(True, alpha=0.3, zorder=0)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS[model_name],
            color="w",
            markerfacecolor=MODEL_COLORS[model_name],
            markeredgecolor="black",
            markersize=12,
            linewidth=0,
            label=MODEL_LABELS[model_name],
        )
        for model_name in MODEL_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(MODEL_ORDER),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=TRADEOFF_LEGEND_FONT_SIZE,
        columnspacing=1.5,
        handletextpad=0.6,
    )
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.16, top=0.78, wspace=0.28)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_seeded_output_path(output_dir: Path, output_name: str, seed: int) -> Path:
    output_path = output_dir / output_name
    return output_path.with_name(f"{output_path.stem}_seed{seed}{output_path.suffix}")


def _plot_calibration_grid(
    all_results: dict[str, dict[str, dict[str, np.ndarray | float]]],
    datasets: list[str],
    output_path: Path,
    *,
    show: bool = False,
) -> None:
    fig, axes = plt.subplots(
        nrows=len(MODEL_ORDER),
        ncols=len(datasets),
        figsize=(4.8 * len(datasets), 3.6 * len(MODEL_ORDER)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for row_idx, model_name in enumerate(MODEL_ORDER):
        for col_idx, dataset_name in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            payload = all_results[dataset_name][model_name]
            y_true = np.asarray(payload["y_true"])
            y_pred = np.asarray(payload["y_pred"])
            y_std = np.asarray(payload["y_std"])

            exp_interval, obs_interval = calibration_curve(
                y_true=y_true,
                y_pred=y_pred,
                y_std=y_std,
                mode="interval",
            )
            mace = calibration_error(y_true, y_pred, y_std, mode="interval")
            mean_log_likelihood = float(payload["mean_log_likelihood"])

            ax.fill_between(
                exp_interval,
                exp_interval,
                obs_interval,
                where=(obs_interval >= exp_interval),
                alpha=0.12,
                color="darkorange",
            )
            ax.fill_between(
                exp_interval,
                exp_interval,
                obs_interval,
                where=(obs_interval < exp_interval),
                alpha=0.12,
                color="steelblue",
            )
            ax.plot(exp_interval, obs_interval, color="black", linewidth=2)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.tick_params(axis="both", which="both", labelsize=CALIBRATION_TICK_FONT_SIZE)
            ax.grid(True, alpha=0.3)

            if row_idx == 0:
                ax.set_title(
                    DATASET_LABELS.get(dataset_name, dataset_name.title()),
                    fontsize=CALIBRATION_TITLE_FONT_SIZE,
                    pad=13,
                )
            if col_idx == 0:
                ax.set_ylabel(
                    f"{MODEL_LABELS[model_name]}\nObserved coverage",
                    fontsize=CALIBRATION_AXIS_LABEL_FONT_SIZE,
                )
            else:
                ax.set_ylabel("Observed coverage", fontsize=CALIBRATION_AXIS_LABEL_FONT_SIZE)
            if row_idx == len(MODEL_ORDER) - 1:
                ax.set_xlabel("Expected coverage", fontsize=CALIBRATION_AXIS_LABEL_FONT_SIZE)

            ax.text(
                0.04,
                0.96,
                f"MACE={mace:.4f}\nMLL={mean_log_likelihood:.4f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=CALIBRATION_TICK_FONT_SIZE,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
            )

    handles = [
        plt.Line2D([0], [0], color="black", linewidth=2, label="Model"),
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Perfect calibration",
        ),
        plt.Rectangle((0, 0), 1, 1, color="darkorange", alpha=0.12, label="Under-confident"),
        plt.Rectangle((0, 0), 1, 1, color="steelblue", alpha=0.12, label="Over-confident"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
        fontsize=CALIBRATION_LEGEND_FONT_SIZE,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _summarize_metrics_across_seeds(summary_df: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        summary_df.groupby(["dataset", "model"], as_index=False)
        .agg(
            num_seeds=("seed", "nunique"),
            mace_interval_mean=("mace_interval", "mean"),
            mace_interval_std=("mace_interval", "std"),
            mean_log_likelihood_mean=("mean_log_likelihood", "mean"),
            mean_log_likelihood_std=("mean_log_likelihood", "std"),
        )
        .sort_values(["dataset", "model"])
        .reset_index(drop=True)
    )
    return aggregated


def _train_and_predict_dataset(
    dataset_name: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, dict[str, np.ndarray | float]]:
    import gpytorch

    _set_seed(seed)
    data_path = _select_dataset_path(
        dataset_name,
        data_path=args.data_path if len(args.datasets) == 1 else None,
        boston_path=args.boston_path,
        concrete_path=args.concrete_path,
        energy_path=args.energy_path,
    )
    X, y = _load_dataset_xy(dataset_name, data_path)
    device = torch.device(args.device)
    dtype = torch.double if args.dtype == "double" else torch.float

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=seed
    )
    train_x_raw = torch.tensor(X_train, dtype=dtype, device=device)
    test_x_raw = torch.tensor(X_test, dtype=dtype, device=device)
    train_y_raw = torch.tensor(y_train, dtype=dtype, device=device)
    test_y_raw = torch.tensor(y_test, dtype=dtype, device=device)
    if train_y_raw.ndim == 1:
        train_y_raw = train_y_raw.unsqueeze(-1)
        test_y_raw = test_y_raw.unsqueeze(-1)

    train_x, test_x, _, _ = _normalize_with_stats(train_x_raw, test_x_raw)
    train_y, test_y, target_mu, target_std = _normalize_with_stats(train_y_raw, test_y_raw)

    is_multitask = train_y.shape[-1] > 1
    num_tasks = train_y.shape[-1]
    output_dim = args.sdr_output_dim

    results: dict[str, dict[str, np.ndarray | float]] = {}

    if is_multitask:
        dkl_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(
            device=device, dtype=dtype
        )
        dkl_model = _MultitaskDKLModel(
            train_x=train_x,
            train_y=train_y,
            likelihood=dkl_likelihood,
            input_dim=train_x.shape[-1],
            embedding_dim=args.dkl_embedding_dim,
            num_tasks=num_tasks,
        ).to(device=device, dtype=dtype)
    else:
        dkl_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
        dkl_model = _DKLModel(
            train_x=train_x,
            train_y=train_y.squeeze(-1),
            likelihood=dkl_likelihood,
            input_dim=train_x.shape[-1],
            embedding_dim=args.dkl_embedding_dim,
        ).to(device=device, dtype=dtype)
    _train_exact_gp(
        model=dkl_model,
        likelihood=dkl_likelihood,
        train_x=train_x,
        train_y=train_y,
        n_iter=args.dkl_iters,
        lr=args.dkl_lr,
        parameter_groups=[
            {"params": dkl_model.feature_extractor.parameters()},
            {"params": dkl_model.covar_module.parameters()},
            {"params": dkl_model.mean_module.parameters()},
            {"params": dkl_model.likelihood.parameters()},
        ],
    )
    dkl_mean, dkl_std = _predict_exact_gp(dkl_model, dkl_likelihood, test_x)
    dkl_metrics = _gp_split_metrics(dkl_model, dkl_likelihood, test_x, test_y)
    if not is_multitask:
        dkl_mean = dkl_mean.unsqueeze(-1)
        dkl_std = dkl_std.unsqueeze(-1)
    dkl_y_true, dkl_y_pred, dkl_y_std, dkl_y_train = _to_original_scale(
        test_y, dkl_mean, dkl_std, target_mu, target_std, train_y
    )
    results["dkl"] = {
        "y_true": dkl_y_true,
        "y_pred": dkl_y_pred,
        "y_std": dkl_y_std,
        "y_train": dkl_y_train,
        "mean_log_likelihood": float(dkl_metrics["mean_log_likelihood"]),
    }

    if is_multitask:
        deep_gp_models, _ = _train_multitask_deep_gp(
            train_x=train_x,
            train_y=train_y,
            n_iter=args.deepgp_iters,
            lr=args.deepgp_lr,
            hidden_dim=args.deepgp_hidden_dim,
            num_inducing=args.deepgp_num_inducing,
            num_likelihood_samples=args.deepgp_likelihood_samples,
        )
        deep_mean, deep_std = _predict_multitask_deep_gp(
            deep_gp_models,
            test_x,
            args.deepgp_likelihood_samples,
        )
        deep_metrics = _multitask_deep_gp_split_metrics(
            deep_gp_models,
            test_x,
            test_y,
            num_likelihood_samples=args.deepgp_likelihood_samples,
        )
    else:
        deep_gp_model = _RegressionDeepGPModel(
            train_x=train_x,
            hidden_dim=args.deepgp_hidden_dim,
            num_inducing=args.deepgp_num_inducing,
        ).to(device=device, dtype=dtype)
        _train_deep_gp(
            model=deep_gp_model,
            train_x=train_x,
            train_y=train_y.squeeze(-1),
            n_iter=args.deepgp_iters,
            lr=args.deepgp_lr,
            num_likelihood_samples=args.deepgp_likelihood_samples,
        )
        deep_mean, deep_std = _predict_deep_gp(
            deep_gp_model,
            test_x,
            args.deepgp_likelihood_samples,
        )
        deep_metrics = _deep_gp_split_metrics(
            deep_gp_model,
            test_x,
            test_y.squeeze(-1),
            num_likelihood_samples=args.deepgp_likelihood_samples,
        )
        deep_mean = deep_mean.unsqueeze(-1)
        deep_std = deep_std.unsqueeze(-1)
    deep_y_true, deep_y_pred, deep_y_std, deep_y_train = _to_original_scale(
        test_y, deep_mean, deep_std, target_mu, target_std, train_y
    )
    results["deep_gp"] = {
        "y_true": deep_y_true,
        "y_pred": deep_y_pred,
        "y_std": deep_y_std,
        "y_train": deep_y_train,
        "mean_log_likelihood": float(deep_metrics["mean_log_likelihood"]),
    }

    train_z_kspca, test_z_kspca = _fit_kernel_supervised_pca_embeddings(
        train_x=train_x,
        test_x=test_x,
        train_y=train_y,
        output_dim=output_dim,
        device=device,
        dtype=dtype,
        kernel=args.kernel_supervised_pca_kernel,
        gamma=args.kernel_supervised_pca_gamma,
        target_kernel=args.kernel_supervised_pca_target_kernel,
        target_gamma=args.kernel_supervised_pca_target_gamma,
        reg=args.kernel_supervised_pca_reg,
    )
    train_z_kspca, test_z_kspca, _, _ = _normalize_with_stats(train_z_kspca, test_z_kspca)
    if is_multitask:
        kspca_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(
            device=device, dtype=dtype
        )
        kspca_model = _MultitaskExactGPModel(
            train_x=train_z_kspca,
            train_y=train_y,
            likelihood=kspca_likelihood,
            ard_num_dims=output_dim,
            num_tasks=num_tasks,
        ).to(device=device, dtype=dtype)
    else:
        kspca_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
        kspca_model = _ExactGPModel(
            train_x=train_z_kspca,
            train_y=train_y.squeeze(-1),
            likelihood=kspca_likelihood,
            ard_num_dims=output_dim,
        ).to(device=device, dtype=dtype)
    _train_exact_gp(
        model=kspca_model,
        likelihood=kspca_likelihood,
        train_x=train_z_kspca,
        train_y=train_y,
        n_iter=args.sdrgp_iters,
        lr=args.sdrgp_lr,
    )
    kspca_mean, kspca_std = _predict_exact_gp(kspca_model, kspca_likelihood, test_z_kspca)
    kspca_metrics = _gp_split_metrics(kspca_model, kspca_likelihood, test_z_kspca, test_y)
    if not is_multitask:
        kspca_mean = kspca_mean.unsqueeze(-1)
        kspca_std = kspca_std.unsqueeze(-1)
    kspca_y_true, kspca_y_pred, kspca_y_std, kspca_y_train = _to_original_scale(
        test_y, kspca_mean, kspca_std, target_mu, target_std, train_y
    )
    results["kernel_supervised_pca_gp"] = {
        "y_true": kspca_y_true,
        "y_pred": kspca_y_pred,
        "y_std": kspca_y_std,
        "y_train": kspca_y_train,
        "mean_log_likelihood": float(kspca_metrics["mean_log_likelihood"]),
    }

    train_z_nca, test_z_nca = _fit_nca_embeddings(
        train_x=train_x,
        test_x=test_x,
        train_y=train_y,
        output_dim=output_dim,
        seed=seed,
        max_iter=args.nca_max_iter,
        tol=args.nca_tol,
        num_bins=args.nca_num_bins,
    )
    train_z_nca, test_z_nca, _, _ = _normalize_with_stats(train_z_nca, test_z_nca)
    if is_multitask:
        nca_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(
            device=device, dtype=dtype
        )
        nca_model = _MultitaskExactGPModel(
            train_x=train_z_nca,
            train_y=train_y,
            likelihood=nca_likelihood,
            ard_num_dims=output_dim,
            num_tasks=num_tasks,
        ).to(device=device, dtype=dtype)
    else:
        nca_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
        nca_model = _ExactGPModel(
            train_x=train_z_nca,
            train_y=train_y.squeeze(-1),
            likelihood=nca_likelihood,
            ard_num_dims=output_dim,
        ).to(device=device, dtype=dtype)
    _train_exact_gp(
        model=nca_model,
        likelihood=nca_likelihood,
        train_x=train_z_nca,
        train_y=train_y,
        n_iter=args.sdrgp_iters,
        lr=args.sdrgp_lr,
    )
    nca_mean, nca_std = _predict_exact_gp(nca_model, nca_likelihood, test_z_nca)
    nca_metrics = _gp_split_metrics(nca_model, nca_likelihood, test_z_nca, test_y)
    if not is_multitask:
        nca_mean = nca_mean.unsqueeze(-1)
        nca_std = nca_std.unsqueeze(-1)
    nca_y_true, nca_y_pred, nca_y_std, nca_y_train = _to_original_scale(
        test_y, nca_mean, nca_std, target_mu, target_std, train_y
    )
    results["nca_gp"] = {
        "y_true": nca_y_true,
        "y_pred": nca_y_pred,
        "y_std": nca_y_std,
        "y_train": nca_y_train,
        "mean_log_likelihood": float(nca_metrics["mean_log_likelihood"]),
    }

    train_z_umap, test_z_umap = _fit_supervised_umap_embeddings(
        train_x=train_x,
        test_x=test_x,
        train_y=train_y,
        output_dim=output_dim,
        seed=seed,
        n_neighbors=args.supervised_umap_n_neighbors,
        min_dist=args.supervised_umap_min_dist,
        target_metric=args.supervised_umap_target_metric,
    )
    train_z_umap, test_z_umap, _, _ = _normalize_with_stats(train_z_umap, test_z_umap)
    if is_multitask:
        umap_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(
            device=device, dtype=dtype
        )
        umap_model = _MultitaskExactGPModel(
            train_x=train_z_umap,
            train_y=train_y,
            likelihood=umap_likelihood,
            ard_num_dims=output_dim,
            num_tasks=num_tasks,
        ).to(device=device, dtype=dtype)
    else:
        umap_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
        umap_model = _ExactGPModel(
            train_x=train_z_umap,
            train_y=train_y.squeeze(-1),
            likelihood=umap_likelihood,
            ard_num_dims=output_dim,
        ).to(device=device, dtype=dtype)
    _train_exact_gp(
        model=umap_model,
        likelihood=umap_likelihood,
        train_x=train_z_umap,
        train_y=train_y,
        n_iter=args.sdrgp_iters,
        lr=args.sdrgp_lr,
    )
    umap_mean, umap_std = _predict_exact_gp(umap_model, umap_likelihood, test_z_umap)
    umap_metrics = _gp_split_metrics(umap_model, umap_likelihood, test_z_umap, test_y)
    if not is_multitask:
        umap_mean = umap_mean.unsqueeze(-1)
        umap_std = umap_std.unsqueeze(-1)
    umap_y_true, umap_y_pred, umap_y_std, umap_y_train = _to_original_scale(
        test_y, umap_mean, umap_std, target_mu, target_std, train_y
    )
    results["supervised_umap_gp"] = {
        "y_true": umap_y_true,
        "y_pred": umap_y_pred,
        "y_std": umap_y_std,
        "y_train": umap_y_train,
        "mean_log_likelihood": float(umap_metrics["mean_log_likelihood"]),
    }

    sdr_kwargs: dict[str, Any] = {
        "affinity_data": SymmetricEntropicAffinity(perp=20, lr=0.01),
        "affinity_embedding": NormalizedGaussianAndStudentAffinity(student=True),
        "hsic_kernel_z": "rq",
        "hsic_kernel_y": "rbf",
        "fgw_loss": "square_loss",
        "loss_fun": "square_loss",
        "alpha": 0.2,
        "output_dim": output_dim,
        "max_iter_outer": 20,
        "tol": 1e-5,
        "lambda_hsic": 0.5,
        "device": device,
        "dtype": dtype,
        "verbose": False,
        "output_sam": train_x.shape[-2],
        "seed": seed,
    }
    dr = SDR(**sdr_kwargs)
    train_z_sdr, _ = dr.fit_transform(train_x, train_y)
    test_z_sdr = dr.transform(test_x)
    train_z_sdr, test_z_sdr, _, _ = _normalize_with_stats(train_z_sdr, test_z_sdr)
    if is_multitask:
        sdr_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(
            device=device, dtype=dtype
        )
        sdr_model = _MultitaskExactGPModel(
            train_x=train_z_sdr,
            train_y=train_y,
            likelihood=sdr_likelihood,
            ard_num_dims=output_dim,
            num_tasks=num_tasks,
        ).to(device=device, dtype=dtype)
    else:
        sdr_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
        sdr_model = _ExactGPModel(
            train_x=train_z_sdr,
            train_y=train_y.squeeze(-1),
            likelihood=sdr_likelihood,
            ard_num_dims=output_dim,
        ).to(device=device, dtype=dtype)
    _train_exact_gp(
        model=sdr_model,
        likelihood=sdr_likelihood,
        train_x=train_z_sdr,
        train_y=train_y,
        n_iter=args.sdrgp_iters,
        lr=args.sdrgp_lr,
    )
    sdr_mean, sdr_std = _predict_exact_gp(sdr_model, sdr_likelihood, test_z_sdr)
    sdr_metrics = _gp_split_metrics(sdr_model, sdr_likelihood, test_z_sdr, test_y)
    if not is_multitask:
        sdr_mean = sdr_mean.unsqueeze(-1)
        sdr_std = sdr_std.unsqueeze(-1)
    sdr_y_true, sdr_y_pred, sdr_y_std, sdr_y_train = _to_original_scale(
        test_y, sdr_mean, sdr_std, target_mu, target_std, train_y
    )
    results["sdrgp"] = {
        "y_true": sdr_y_true,
        "y_pred": sdr_y_pred,
        "y_std": sdr_y_std,
        "y_train": sdr_y_train,
        "mean_log_likelihood": float(sdr_metrics["mean_log_likelihood"]),
    }

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the five GP regression benchmark models across one or more random seeds, "
            "save one interval calibration grid per seed, and plot the MLL-vs-MACE trade-off "
            "from seed-averaged metrics."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_ORDER),
        choices=list(DATASET_ORDER),
        help="Datasets to include as columns in the final figure.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional single random seed for backward compatibility. Ignored if --seeds is passed.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of random seeds. If omitted, the script uses five seeds: 0 1 2 3 4.",
    )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--boston-path", type=str, default=None)
    parser.add_argument("--concrete-path", type=str, default=None)
    parser.add_argument("--energy-path", type=str, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--dkl-iters", type=int, default=20)
    parser.add_argument("--sdrgp-iters", type=int, default=20)
    parser.add_argument("--dkl-lr", type=float, default=0.1)
    parser.add_argument("--sdrgp-lr", type=float, default=0.1)
    parser.add_argument("--dkl-embedding-dim", type=int, default=2)
    parser.add_argument("--sdr-output-dim", type=int, default=2)
    parser.add_argument("--deepgp-iters", type=int, default=20)
    parser.add_argument("--deepgp-lr", type=float, default=0.1)
    parser.add_argument("--deepgp-hidden-dim", type=int, default=2)
    parser.add_argument("--deepgp-num-inducing", type=int, default=128)
    parser.add_argument("--deepgp-likelihood-samples", type=int, default=16)
    parser.add_argument("--supervised-umap-n-neighbors", type=int, default=50)
    parser.add_argument("--supervised-umap-min-dist", type=float, default=0.1)
    parser.add_argument("--supervised-umap-target-metric", type=str, default="l2")
    parser.add_argument("--kernel-supervised-pca-kernel", type=str, default="rbf")
    parser.add_argument("--kernel-supervised-pca-gamma", type=float, default=None)
    parser.add_argument("--kernel-supervised-pca-target-kernel", type=str, default="auto")
    parser.add_argument("--kernel-supervised-pca-target-gamma", type=float, default=None)
    parser.add_argument("--kernel-supervised-pca-reg", type=float, default=1e-6)
    parser.add_argument("--nca-max-iter", type=int, default=100)
    parser.add_argument("--nca-tol", type=float, default=1e-5)
    parser.add_argument("--nca-num-bins", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, choices=("float", "double"), default="double")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_outputs/calibration_results",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="gp_interval_calibration_grid.pdf",
    )
    parser.add_argument(
        "--tradeoff-output-name",
        type=str,
        default="gp_mll_mace_tradeoff.pdf",
        help="Filename for the MLL-vs-MACE trade-off figure.",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure after saving it.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.seeds is not None:
        seeds = list(args.seeds)
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(range(5))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | str]] = []
    saved_calibration_paths: list[Path] = []

    for seed in seeds:
        all_results: dict[str, dict[str, dict[str, np.ndarray | float]]] = {}
        for dataset_name in args.datasets:
            dataset_results = _train_and_predict_dataset(dataset_name, seed, args)
            all_results[dataset_name] = dataset_results
            for model_name, payload in dataset_results.items():
                y_true = np.asarray(payload["y_true"])
                y_pred = np.asarray(payload["y_pred"])
                y_std = np.asarray(payload["y_std"])
                mace = calibration_error(y_true, y_pred, y_std, mode="interval")
                mean_log_likelihood = float(payload["mean_log_likelihood"])
                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "model": model_name,
                        "mace_interval": float(mace),
                        "mean_log_likelihood": mean_log_likelihood,
                    }
                )

        calibration_output_path = _build_seeded_output_path(output_dir, args.output_name, seed)
        _plot_calibration_grid(
            all_results,
            list(args.datasets),
            calibration_output_path,
            show=args.show,
        )
        saved_calibration_paths.append(calibration_output_path)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "gp_interval_calibration_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    aggregate_summary_df = _summarize_metrics_across_seeds(summary_df)
    aggregate_summary_path = output_dir / "gp_interval_calibration_metrics_summary.csv"
    aggregate_summary_df.to_csv(aggregate_summary_path, index=False)
    tradeoff_output_path = output_dir / args.tradeoff_output_name
    _plot_mll_vs_mace_tradeoff(aggregate_summary_df, list(args.datasets), tradeoff_output_path)

    for calibration_output_path in saved_calibration_paths:
        print(f"Saved calibration figure to {calibration_output_path}")
    print(f"Saved trade-off figure to {tradeoff_output_path}")
    print(f"Saved metrics to {summary_path}")
    print(f"Saved aggregated metrics to {aggregate_summary_path}")
    print(summary_df.to_string(index=False))
    print(aggregate_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
