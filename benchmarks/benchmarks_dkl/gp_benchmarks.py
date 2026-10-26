import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import gpytorch
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NeighborhoodComponentsAnalysis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.affinities import NormalizedGaussianAndStudentAffinity, SymmetricEntropicAffinity
from src.kernel_supervised_pca import KernelSupervisedPCA
from src.sdr import SDR


SUPPORTED_DATASETS = ("boston", "concrete", "energy")
SUPPORTED_METHODS = (
    "dkl",
    "deep_gp",
    "kernel_supervised_pca_gp",
    "nca_gp",
    "supervised_umap_gp",
    "sdrgp",
)


def _normalize_requested_methods(
    methods: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if methods is None:
        return SUPPORTED_METHODS
    normalized = tuple(str(method).lower() for method in methods)
    invalid = sorted(set(normalized) - set(SUPPORTED_METHODS))
    if invalid:
        raise ValueError(f"Unknown methods {invalid}. Supported methods: {list(SUPPORTED_METHODS)}")
    # Preserve caller order while dropping duplicates.
    return tuple(dict.fromkeys(normalized))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_dataset_path(dataset_name: str, data_path: str | Path | None) -> Path:
    dataset_name = dataset_name.lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: {SUPPORTED_DATASETS}")

    if data_path is not None:
        candidate = Path(data_path)
        if candidate.exists():
            return candidate
    default_files = {
        "boston": "boston.csv.xls",
        "concrete": "Concrete_Data.xls",
        "energy": "ENB2012_data.xlsx",
    }
    default_rel = Path("data/regression_datasets") / default_files[dataset_name]
    if default_rel.exists():
        return default_rel
    module_root = Path(__file__).resolve().parent.parent
    module_default = module_root / "data" / "regression_datasets" / default_files[dataset_name]
    if module_default.exists():
        return module_default
    raise FileNotFoundError(f"Could not locate {dataset_name} dataset. Set data_path explicitly.")


def _select_dataset_path(
    dataset_name: str,
    data_path: str | Path | None = None,
    boston_path: str | Path | None = None,
    concrete_path: str | Path | None = None,
    energy_path: str | Path | None = None,
) -> str | Path | None:
    dataset_paths = {
        "boston": boston_path,
        "concrete": concrete_path,
        "energy": energy_path,
    }
    return dataset_paths.get(dataset_name.lower(), data_path) or data_path


def _load_dataset_xy(
    dataset_name: str, data_path: str | Path | None
) -> tuple[np.ndarray, np.ndarray]:
    dataset_name = dataset_name.lower()
    path = _resolve_dataset_path(dataset_name, data_path)
    if dataset_name == "boston":
        df = pd.read_csv(path)
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        return X, y
    if dataset_name == "concrete":
        df = pd.read_excel(path)
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        # # Match existing notebook preprocessing.
        # X[:, -1] = np.log(X[:, -1])
        return X, y
    if dataset_name == "energy":
        df = pd.read_excel(path)
        X = df.iloc[:, :-2].to_numpy()
        y = df.iloc[:, -2:].to_numpy()
        return X, y
    raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: {SUPPORTED_DATASETS}")


def _normalize_with_train_stats(
    train_t: torch.Tensor, test_t: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    mu = train_t.mean(0, keepdim=True)
    std = train_t.std(0, keepdim=True).clamp_min(eps)
    return (train_t - mu) / std, (test_t - mu) / std


def _flatten_targets(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[-1] == 1:
        return y.squeeze(-1)
    return y


def _gp_split_metrics(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods._GaussianLikelihoodBase,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, float]:
    model.eval()
    likelihood.eval()
    y_ = _flatten_targets(y)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x))
        mean = pred.mean
        mse = torch.mean((mean - y_) ** 2)
        var = pred.variance.clamp_min(eps)
        if mean.ndim == 1:
            dist = torch.distributions.Normal(mean, var.sqrt())
            mean_log_lik = dist.log_prob(y_).mean()
        else:
            try:
                mean_log_lik = pred.log_prob(y_) / y_.numel()
            except Exception:
                # Fall back to independent Gaussian marginals when the full multitask
                # predictive covariance is numerically not PSD enough for log_prob().
                dist = torch.distributions.Normal(mean, var.sqrt())
                mean_log_lik = dist.log_prob(y_).mean()
    return {"mse": float(mse.item()), "mean_log_likelihood": float(mean_log_lik.item())}


class _FeatureExtractor(torch.nn.Sequential):
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.add_module("linear1", torch.nn.Linear(input_dim, 1000))
        self.add_module("relu1", torch.nn.ReLU())
        self.add_module("linear2", torch.nn.Linear(1000, 500))
        self.add_module("relu2", torch.nn.ReLU())
        self.add_module("linear3", torch.nn.Linear(500, 50))
        self.add_module("relu3", torch.nn.ReLU())
        self.add_module("linear4", torch.nn.Linear(50, embedding_dim))


class _DKLModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        input_dim: int,
        embedding_dim: int,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = _FeatureExtractor(input_dim, embedding_dim)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=embedding_dim)
        )
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        projected_x = self.scale_to_bounds(self.feature_extractor(x))
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class _ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        ard_num_dims: int,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class _MultitaskDKLModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
        input_dim: int,
        embedding_dim: int,
        num_tasks: int,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = _FeatureExtractor(input_dim, embedding_dim)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=embedding_dim)),
            num_tasks=num_tasks,
            rank=num_tasks,
        )
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultitaskMultivariateNormal:
        projected_x = self.scale_to_bounds(self.feature_extractor(x))
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class _MultitaskExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
        ard_num_dims: int,
        num_tasks: int,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)),
            num_tasks=num_tasks,
            rank=num_tasks,
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultitaskMultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class _DeepGPHiddenLayer(gpytorch.models.deep_gps.DeepGPLayer):
    def __init__(
        self,
        input_dims: int,
        output_dims: int | None,
        inducing_points: torch.Tensor,
    ):
        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2),
            batch_shape=batch_shape,
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy, input_dims=input_dims, output_dims=output_dims)
        if output_dims is None:
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            self.mean_module = gpytorch.means.LinearMean(input_dims)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dims, batch_shape=batch_shape),
            batch_shape=batch_shape,
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class _RegressionDeepGPModel(gpytorch.models.deep_gps.DeepGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        hidden_dim: int,
        num_inducing: int,
    ):
        super().__init__()
        hidden_inducing = _make_inducing_points(
            train_x=train_x,
            num_inducing=num_inducing,
            batch_size=hidden_dim,
        )
        output_inducing = torch.randn(
            min(num_inducing, train_x.shape[-2]),
            hidden_dim,
            device=train_x.device,
            dtype=train_x.dtype,
        )
        self.hidden_layer = _DeepGPHiddenLayer(
            input_dims=train_x.shape[-1],
            output_dims=hidden_dim,
            inducing_points=hidden_inducing,
        )
        self.output_layer = _DeepGPHiddenLayer(
            input_dims=hidden_dim,
            output_dims=None,
            inducing_points=output_inducing,
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        hidden_representation = self.hidden_layer(x)
        return self.output_layer(hidden_representation)


def _kernel_params(model: gpytorch.models.ExactGP) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        out["outputscale"] = float(model.covar_module.outputscale.item())
    except Exception:
        try:
            out["outputscale"] = float(model.covar_module.data_covar_module.outputscale.item())
        except Exception:
            pass
    try:
        ls = model.covar_module.base_kernel.lengthscale.detach().mean()
        out["lengthscale_mean"] = float(ls.item())
    except Exception:
        try:
            ls = model.covar_module.base_kernel.base_kernel.lengthscale.detach().mean()
            out["lengthscale_mean"] = float(ls.item())
        except Exception:
            pass
    try:
        out["task_covar_trace"] = float(
            model.covar_module.task_covar_module.covar_matrix.trace().item()
        )
    except Exception:
        pass
    return out


def _likelihood_noise_value(likelihood: gpytorch.likelihoods._GaussianLikelihoodBase) -> float:
    try:
        return float(likelihood.noise.mean().item())
    except Exception:
        try:
            return float(likelihood.task_noises.mean().item())
        except Exception:
            return float("nan")


def _make_inducing_points(
    train_x: torch.Tensor,
    num_inducing: int,
    batch_size: int | None = None,
) -> torch.Tensor:
    num_inducing = max(1, min(int(num_inducing), train_x.shape[-2]))
    permutation = torch.randperm(train_x.shape[-2], device=train_x.device)[:num_inducing]
    inducing_points = train_x.index_select(0, permutation).clone()
    if batch_size is None:
        return inducing_points
    return inducing_points.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def _mean_over_leading_dims(values: torch.Tensor, target_ndim: int) -> torch.Tensor:
    reduced = values
    while reduced.ndim > target_ndim:
        reduced = reduced.mean(dim=0)
    return reduced


def _train_exact_gp(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods._GaussianLikelihoodBase,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_iter: int,
    lr: float,
    parameter_groups: list[dict[str, Any]] | None = None,
) -> list[dict[str, float]]:
    model.train()
    likelihood.train()
    optimizer = (
        torch.optim.Adam(parameter_groups, lr=lr)
        if parameter_groups is not None
        else torch.optim.Adam(model.parameters(), lr=lr)
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    logs: list[dict[str, float]] = []
    target = _flatten_targets(train_y)
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        neg_mll = -mll(output, target)
        neg_mll.backward()
        optimizer.step()

        step = {
            "iter": float(i + 1),
            "neg_mll": float(neg_mll.item()),
            "mll": float((-neg_mll).item()),
            "noise": _likelihood_noise_value(likelihood),
        }
        step.update(_kernel_params(model))
        logs.append(step)
    return logs


def _train_deep_gp(
    model: _RegressionDeepGPModel,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_iter: int,
    lr: float,
    num_likelihood_samples: int,
) -> list[dict[str, float]]:
    model.train()
    model.likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.DeepApproximateMLL(
        gpytorch.mlls.VariationalELBO(
            model.likelihood,
            model,
            num_data=train_x.shape[-2],
        )
    )

    logs: list[dict[str, float]] = []
    target = _flatten_targets(train_y)
    for i in range(n_iter):
        optimizer.zero_grad()
        with gpytorch.settings.num_likelihood_samples(num_likelihood_samples):
            output = model(train_x)
            neg_elbo = -mll(output, target)
        neg_elbo.backward()
        optimizer.step()
        logs.append(
            {
                "iter": float(i + 1),
                "neg_elbo": float(neg_elbo.item()),
                "elbo": float((-neg_elbo).item()),
            }
        )
    return logs


def _deep_gp_split_metrics(
    model: _RegressionDeepGPModel,
    x: torch.Tensor,
    y: torch.Tensor,
    num_likelihood_samples: int,
    eps: float = 1e-6,
) -> dict[str, float]:
    model.eval()
    model.likelihood.eval()
    y_ = _flatten_targets(y)
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(num_likelihood_samples):
        pred = model.likelihood(model(x))
        mean = _mean_over_leading_dims(pred.mean, target_ndim=1)
        variance = _mean_over_leading_dims(pred.variance, target_ndim=1).clamp_min(eps)
        mse = torch.mean((mean - y_) ** 2)
        mean_log_lik = torch.distributions.Normal(mean, variance.sqrt()).log_prob(y_).mean()
    return {"mse": float(mse.item()), "mean_log_likelihood": float(mean_log_lik.item())}


def _train_multitask_deep_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_iter: int,
    lr: float,
    hidden_dim: int,
    num_inducing: int,
    num_likelihood_samples: int,
) -> tuple[list[_RegressionDeepGPModel], list[dict[str, Any]]]:
    models: list[_RegressionDeepGPModel] = []
    logs: list[dict[str, Any]] = []
    for task_idx in range(train_y.shape[-1]):
        model = _RegressionDeepGPModel(
            train_x=train_x,
            hidden_dim=hidden_dim,
            num_inducing=num_inducing,
        ).to(device=train_x.device, dtype=train_x.dtype)
        task_logs = _train_deep_gp(
            model=model,
            train_x=train_x,
            train_y=train_y[:, task_idx],
            n_iter=n_iter,
            lr=lr,
            num_likelihood_samples=num_likelihood_samples,
        )
        models.append(model)
        logs.append({"task": float(task_idx), "logs": task_logs})
    return models, logs


def _multitask_deep_gp_split_metrics(
    models: list[_RegressionDeepGPModel],
    x: torch.Tensor,
    y: torch.Tensor,
    num_likelihood_samples: int,
    eps: float = 1e-6,
) -> dict[str, float]:
    means: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    for model in models:
        model.eval()
        model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(num_likelihood_samples):
            pred = model.likelihood(model(x))
            means.append(_mean_over_leading_dims(pred.mean, target_ndim=1))
            variances.append(_mean_over_leading_dims(pred.variance, target_ndim=1).clamp_min(eps))
    mean = torch.stack(means, dim=-1)
    variance = torch.stack(variances, dim=-1)
    mse = torch.mean((mean - y) ** 2)
    mean_log_lik = torch.distributions.Normal(mean, variance.sqrt()).log_prob(y).mean()
    return {"mse": float(mse.item()), "mean_log_likelihood": float(mean_log_lik.item())}


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


def _make_nca_supervision_targets(
    train_y: torch.Tensor,
    num_bins: int,
) -> np.ndarray:
    train_y_np = train_y.detach().cpu().numpy()
    if train_y_np.ndim == 1 or (train_y_np.ndim == 2 and train_y_np.shape[-1] == 1):
        signal = np.asarray(train_y_np).reshape(-1)
    else:
        signal = np.asarray(train_y_np).mean(axis=-1)

    num_samples = signal.shape[0]
    if num_samples <= 1:
        return np.zeros(num_samples, dtype=np.int64)

    num_bins = max(2, min(int(num_bins), num_samples))
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(signal, quantiles)
    edges = np.unique(edges)
    if edges.size <= 2:
        ranks = np.argsort(np.argsort(signal, kind="mergesort"), kind="mergesort")
        return (
            np.floor(ranks * num_bins / max(num_samples, 1)).astype(np.int64).clip(0, num_bins - 1)
        )

    interior_edges = edges[1:-1]
    return np.digitize(signal, interior_edges, right=False).astype(np.int64)


def _fit_nca_embeddings(
    train_x: torch.Tensor,
    test_x: torch.Tensor,
    train_y: torch.Tensor,
    output_dim: int,
    seed: int,
    max_iter: int,
    tol: float,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    nca = NeighborhoodComponentsAnalysis(
        n_components=output_dim,
        init="pca",
        random_state=seed,
        max_iter=max_iter,
        tol=tol,
    )
    supervision_targets = _make_nca_supervision_targets(train_y, num_bins=num_bins)
    train_z_np = nca.fit_transform(train_x.detach().cpu().numpy(), supervision_targets)
    test_z_np = nca.transform(test_x.detach().cpu().numpy())
    train_z = torch.as_tensor(train_z_np, device=train_x.device, dtype=train_x.dtype)
    test_z = torch.as_tensor(test_z_np, device=test_x.device, dtype=test_x.dtype)
    return train_z, test_z


def _summarize_metrics_across_seeds(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    metric_columns = [
        column
        for column in metrics_df.columns
        if column not in {"dataset", "seed", "model"}
        and pd.api.types.is_numeric_dtype(metrics_df[column])
    ]
    if not metric_columns:
        return pd.DataFrame()

    summary_df = (
        metrics_df.groupby(["dataset", "model"], as_index=False)[metric_columns]
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
    summary_df.insert(2, "num_seeds", metrics_df.groupby(["dataset", "model"]).size().to_numpy())
    return summary_df


def _split_summary_by_model(summary_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if summary_df.empty or "model" not in summary_df.columns:
        return {}
    return {
        str(model_name): model_df.reset_index(drop=True)
        for model_name, model_df in summary_df.groupby("model", sort=False)
    }


def benchmark_dkl_vs_sdrgp_boston(
    seeds: list[int] | tuple[int, ...],
    methods: list[str] | tuple[str, ...] | None = None,
    data_path: str | Path | None = None,
    boston_path: str | Path | None = None,
    test_size: float = 0.2,
    dkl_iters: int = 20,
    sdrgp_iters: int = 20,
    dkl_lr: float = 0.01,
    sdrgp_lr: float = 0.1,
    dkl_embedding_dim: int = 3,
    sdr_output_dim: int = 3,
    deepgp_iters: int = 50,
    deepgp_lr: float = 0.01,
    deepgp_hidden_dim: int = 4,
    deepgp_num_inducing: int = 128,
    deepgp_likelihood_samples: int = 16,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.double,
    sdr_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Benchmark DKL vs GP-on-SDR embeddings across random seeds on Boston Housing.

    Returns a dict containing:
    - per_seed_results: list of full logs/metrics/runtimes for each seed
    - metrics_df: per-seed tabular metrics for both models
    - runtime_df: per-seed tabular runtimes
    """

    return benchmark_dkl_vs_sdrgp_dataset(
        dataset_name="boston",
        seeds=seeds,
        methods=methods,
        data_path=_select_dataset_path("boston", data_path=data_path, boston_path=boston_path),
        test_size=test_size,
        dkl_iters=dkl_iters,
        sdrgp_iters=sdrgp_iters,
        dkl_lr=dkl_lr,
        sdrgp_lr=sdrgp_lr,
        dkl_embedding_dim=dkl_embedding_dim,
        sdr_output_dim=sdr_output_dim,
        deepgp_iters=deepgp_iters,
        deepgp_lr=deepgp_lr,
        deepgp_hidden_dim=deepgp_hidden_dim,
        deepgp_num_inducing=deepgp_num_inducing,
        deepgp_likelihood_samples=deepgp_likelihood_samples,
        device=device,
        dtype=dtype,
        sdr_overrides=sdr_overrides,
    )


def benchmark_dkl_vs_sdrgp_energy(
    seeds: list[int] | tuple[int, ...],
    methods: list[str] | tuple[str, ...] | None = None,
    data_path: str | Path | None = None,
    energy_path: str | Path | None = None,
    test_size: float = 0.2,
    dkl_iters: int = 20,
    sdrgp_iters: int = 20,
    dkl_lr: float = 0.01,
    sdrgp_lr: float = 0.1,
    dkl_embedding_dim: int = 3,
    sdr_output_dim: int = 3,
    deepgp_iters: int = 50,
    deepgp_lr: float = 0.01,
    deepgp_hidden_dim: int = 4,
    deepgp_num_inducing: int = 128,
    deepgp_likelihood_samples: int = 16,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.double,
    sdr_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return benchmark_dkl_vs_sdrgp_dataset(
        dataset_name="energy",
        seeds=seeds,
        methods=methods,
        data_path=_select_dataset_path("energy", data_path=data_path, energy_path=energy_path),
        test_size=test_size,
        dkl_iters=dkl_iters,
        sdrgp_iters=sdrgp_iters,
        dkl_lr=dkl_lr,
        sdrgp_lr=sdrgp_lr,
        dkl_embedding_dim=dkl_embedding_dim,
        sdr_output_dim=sdr_output_dim,
        deepgp_iters=deepgp_iters,
        deepgp_lr=deepgp_lr,
        deepgp_hidden_dim=deepgp_hidden_dim,
        deepgp_num_inducing=deepgp_num_inducing,
        deepgp_likelihood_samples=deepgp_likelihood_samples,
        device=device,
        dtype=dtype,
        sdr_overrides=sdr_overrides,
    )


def benchmark_dkl_vs_sdrgp_dataset(
    dataset_name: str,
    seeds: list[int] | tuple[int, ...],
    methods: list[str] | tuple[str, ...] | None = None,
    data_path: str | Path | None = None,
    boston_path: str | Path | None = None,
    concrete_path: str | Path | None = None,
    energy_path: str | Path | None = None,
    test_size: float = 0.2,
    dkl_iters: int = 20,
    sdrgp_iters: int = 20,
    dkl_lr: float = 0.01,
    sdrgp_lr: float = 0.1,
    dkl_embedding_dim: int = 3,
    sdr_output_dim: int = 3,
    supervised_umap_n_neighbors: int = 50,
    supervised_umap_min_dist: float = 0.1,
    supervised_umap_target_metric: str = "l2",
    kernel_supervised_pca_kernel: str = "rbf",
    kernel_supervised_pca_gamma: float | None = None,
    kernel_supervised_pca_target_kernel: str = "auto",
    kernel_supervised_pca_target_gamma: float | None = None,
    kernel_supervised_pca_reg: float = 1e-6,
    nca_max_iter: int = 100,
    nca_tol: float = 1e-5,
    nca_num_bins: int = 10,
    deepgp_iters: int = 50,
    deepgp_lr: float = 0.01,
    deepgp_hidden_dim: int = 4,
    deepgp_num_inducing: int = 128,
    deepgp_likelihood_samples: int = 16,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.double,
    sdr_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_name = dataset_name.lower()
    selected_methods = _normalize_requested_methods(methods)
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

    base_sdr_kwargs: dict[str, Any] = {
        "affinity_data": SymmetricEntropicAffinity(perp=20, lr=0.01),
        "affinity_embedding": NormalizedGaussianAndStudentAffinity(student=True),
        "hsic_kernel_z": "rq",
        "hsic_kernel_y": "rbf",
        "fgw_loss": "square_loss",
        "loss_fun": "square_loss",
        "alpha": 0.2,
        "output_dim": sdr_output_dim,
        "max_iter_outer": 20,
        "tol": 1e-5,
        "lambda_hsic": 0.5,
        "device": device,
        "dtype": dtype,
        "verbose": False,
    }
    if sdr_overrides:
        base_sdr_kwargs.update(sdr_overrides)

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

        seed_metrics: dict[str, dict[str, dict[str, float]]] = {}
        seed_logs: dict[str, Any] = {}
        seed_runtimes: dict[str, float] = {}

        if "dkl" in selected_methods:
            if is_multitask:
                dkl_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=num_tasks
                ).to(device=device, dtype=dtype)
                dkl_model = _MultitaskDKLModel(
                    train_x=train_x,
                    train_y=train_y,
                    likelihood=dkl_likelihood,
                    input_dim=train_x.shape[-1],
                    embedding_dim=dkl_embedding_dim,
                    num_tasks=num_tasks,
                ).to(device=device, dtype=dtype)
            else:
                dkl_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
                    device=device, dtype=dtype
                )
                dkl_model = _DKLModel(
                    train_x=train_x,
                    train_y=train_y.squeeze(-1),
                    likelihood=dkl_likelihood,
                    input_dim=train_x.shape[-1],
                    embedding_dim=dkl_embedding_dim,
                ).to(device=device, dtype=dtype)
            dkl_param_groups = [
                {"params": dkl_model.feature_extractor.parameters()},
                {"params": dkl_model.covar_module.parameters()},
                {"params": dkl_model.mean_module.parameters()},
                {"params": dkl_model.likelihood.parameters()},
            ]

            t0 = time.perf_counter()
            dkl_logs = _train_exact_gp(
                model=dkl_model,
                likelihood=dkl_likelihood,
                train_x=train_x,
                train_y=train_y,
                n_iter=dkl_iters,
                lr=dkl_lr,
                parameter_groups=dkl_param_groups,
            )
            seed_runtimes["dkl_train"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            dkl_train_metrics = _gp_split_metrics(dkl_model, dkl_likelihood, train_x, train_y)
            dkl_test_metrics = _gp_split_metrics(dkl_model, dkl_likelihood, test_x, test_y)
            seed_runtimes["dkl_eval"] = time.perf_counter() - t0

            seed_metrics["dkl"] = {"train": dkl_train_metrics, "test": dkl_test_metrics}
            seed_logs["dkl"] = dkl_logs

        if "deep_gp" in selected_methods:
            t0 = time.perf_counter()
            if is_multitask:
                deep_gp_models, deep_gp_logs = _train_multitask_deep_gp(
                    train_x=train_x,
                    train_y=train_y,
                    n_iter=deepgp_iters,
                    lr=deepgp_lr,
                    hidden_dim=deepgp_hidden_dim,
                    num_inducing=deepgp_num_inducing,
                    num_likelihood_samples=deepgp_likelihood_samples,
                )
            else:
                deep_gp_model = _RegressionDeepGPModel(
                    train_x=train_x,
                    hidden_dim=deepgp_hidden_dim,
                    num_inducing=deepgp_num_inducing,
                ).to(device=device, dtype=dtype)
                deep_gp_logs = _train_deep_gp(
                    model=deep_gp_model,
                    train_x=train_x,
                    train_y=train_y.squeeze(-1),
                    n_iter=deepgp_iters,
                    lr=deepgp_lr,
                    num_likelihood_samples=deepgp_likelihood_samples,
                )
            seed_runtimes["deep_gp_train"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            if is_multitask:
                deep_gp_train_metrics = _multitask_deep_gp_split_metrics(
                    deep_gp_models,
                    train_x,
                    train_y,
                    num_likelihood_samples=deepgp_likelihood_samples,
                )
                deep_gp_test_metrics = _multitask_deep_gp_split_metrics(
                    deep_gp_models,
                    test_x,
                    test_y,
                    num_likelihood_samples=deepgp_likelihood_samples,
                )
            else:
                deep_gp_train_metrics = _deep_gp_split_metrics(
                    deep_gp_model,
                    train_x,
                    train_y.squeeze(-1),
                    num_likelihood_samples=deepgp_likelihood_samples,
                )
                deep_gp_test_metrics = _deep_gp_split_metrics(
                    deep_gp_model,
                    test_x,
                    test_y.squeeze(-1),
                    num_likelihood_samples=deepgp_likelihood_samples,
                )
            seed_runtimes["deep_gp_eval"] = time.perf_counter() - t0

            seed_metrics["deep_gp"] = {
                "train": deep_gp_train_metrics,
                "test": deep_gp_test_metrics,
            }
            seed_logs["deep_gp"] = deep_gp_logs

        if "kernel_supervised_pca_gp" in selected_methods:
            t0 = time.perf_counter()
            train_z_kspca, test_z_kspca = _fit_kernel_supervised_pca_embeddings(
                train_x=train_x,
                test_x=test_x,
                train_y=train_y,
                output_dim=sdr_output_dim,
                device=device,
                dtype=dtype,
                kernel=kernel_supervised_pca_kernel,
                gamma=kernel_supervised_pca_gamma,
                target_kernel=kernel_supervised_pca_target_kernel,
                target_gamma=kernel_supervised_pca_target_gamma,
                reg=kernel_supervised_pca_reg,
            )
            seed_runtimes["kernel_supervised_pca_fit"] = time.perf_counter() - t0

            train_z_kspca, test_z_kspca = _normalize_with_train_stats(train_z_kspca, test_z_kspca)

            if is_multitask:
                kspca_gp_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=num_tasks
                ).to(device=device, dtype=dtype)
                kspca_gp_model = _MultitaskExactGPModel(
                    train_x=train_z_kspca,
                    train_y=train_y,
                    likelihood=kspca_gp_likelihood,
                    ard_num_dims=sdr_output_dim,
                    num_tasks=num_tasks,
                ).to(device=device, dtype=dtype)
            else:
                kspca_gp_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
                    device=device, dtype=dtype
                )
                kspca_gp_model = _ExactGPModel(
                    train_x=train_z_kspca,
                    train_y=train_y.squeeze(-1),
                    likelihood=kspca_gp_likelihood,
                    ard_num_dims=sdr_output_dim,
                ).to(device=device, dtype=dtype)

            t0 = time.perf_counter()
            kspca_gp_logs = _train_exact_gp(
                model=kspca_gp_model,
                likelihood=kspca_gp_likelihood,
                train_x=train_z_kspca,
                train_y=train_y,
                n_iter=sdrgp_iters,
                lr=sdrgp_lr,
            )
            seed_runtimes["kernel_supervised_pca_gp_train"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            kspca_gp_train_metrics = _gp_split_metrics(
                kspca_gp_model, kspca_gp_likelihood, train_z_kspca, train_y
            )
            kspca_gp_test_metrics = _gp_split_metrics(
                kspca_gp_model, kspca_gp_likelihood, test_z_kspca, test_y
            )
            seed_runtimes["kernel_supervised_pca_gp_eval"] = time.perf_counter() - t0

            seed_metrics["kernel_supervised_pca_gp"] = {
                "train": kspca_gp_train_metrics,
                "test": kspca_gp_test_metrics,
            }
            seed_logs["kernel_supervised_pca_gp"] = kspca_gp_logs

        if "nca_gp" in selected_methods:
            t0 = time.perf_counter()
            train_z_nca, test_z_nca = _fit_nca_embeddings(
                train_x=train_x,
                test_x=test_x,
                train_y=train_y,
                output_dim=sdr_output_dim,
                seed=int(seed),
                max_iter=nca_max_iter,
                tol=nca_tol,
                num_bins=nca_num_bins,
            )
            seed_runtimes["nca_fit"] = time.perf_counter() - t0

            train_z_nca, test_z_nca = _normalize_with_train_stats(train_z_nca, test_z_nca)

            if is_multitask:
                nca_gp_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=num_tasks
                ).to(device=device, dtype=dtype)
                nca_gp_model = _MultitaskExactGPModel(
                    train_x=train_z_nca,
                    train_y=train_y,
                    likelihood=nca_gp_likelihood,
                    ard_num_dims=sdr_output_dim,
                    num_tasks=num_tasks,
                ).to(device=device, dtype=dtype)
            else:
                nca_gp_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
                    device=device, dtype=dtype
                )
                nca_gp_model = _ExactGPModel(
                    train_x=train_z_nca,
                    train_y=train_y.squeeze(-1),
                    likelihood=nca_gp_likelihood,
                    ard_num_dims=sdr_output_dim,
                ).to(device=device, dtype=dtype)

            t0 = time.perf_counter()
            nca_gp_logs = _train_exact_gp(
                model=nca_gp_model,
                likelihood=nca_gp_likelihood,
                train_x=train_z_nca,
                train_y=train_y,
                n_iter=sdrgp_iters,
                lr=sdrgp_lr,
            )
            seed_runtimes["nca_gp_train"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            nca_gp_train_metrics = _gp_split_metrics(
                nca_gp_model, nca_gp_likelihood, train_z_nca, train_y
            )
            nca_gp_test_metrics = _gp_split_metrics(
                nca_gp_model, nca_gp_likelihood, test_z_nca, test_y
            )
            seed_runtimes["nca_gp_eval"] = time.perf_counter() - t0

            seed_metrics["nca_gp"] = {"train": nca_gp_train_metrics, "test": nca_gp_test_metrics}
            seed_logs["nca_gp"] = nca_gp_logs

        if "supervised_umap_gp" in selected_methods:
            t0 = time.perf_counter()
            train_z_umap, test_z_umap = _fit_supervised_umap_embeddings(
                train_x=train_x,
                test_x=test_x,
                train_y=train_y,
                output_dim=sdr_output_dim,
                seed=int(seed),
                n_neighbors=supervised_umap_n_neighbors,
                min_dist=supervised_umap_min_dist,
                target_metric=supervised_umap_target_metric,
            )
            seed_runtimes["supervised_umap_fit"] = time.perf_counter() - t0

            train_z_umap, test_z_umap = _normalize_with_train_stats(train_z_umap, test_z_umap)

            if is_multitask:
                supervised_umap_gp_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=num_tasks
                ).to(device=device, dtype=dtype)
                supervised_umap_gp_model = _MultitaskExactGPModel(
                    train_x=train_z_umap,
                    train_y=train_y,
                    likelihood=supervised_umap_gp_likelihood,
                    ard_num_dims=sdr_output_dim,
                    num_tasks=num_tasks,
                ).to(device=device, dtype=dtype)
            else:
                supervised_umap_gp_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
                    device=device, dtype=dtype
                )
                supervised_umap_gp_model = _ExactGPModel(
                    train_x=train_z_umap,
                    train_y=train_y.squeeze(-1),
                    likelihood=supervised_umap_gp_likelihood,
                    ard_num_dims=sdr_output_dim,
                ).to(device=device, dtype=dtype)

            t0 = time.perf_counter()
            supervised_umap_gp_logs = _train_exact_gp(
                model=supervised_umap_gp_model,
                likelihood=supervised_umap_gp_likelihood,
                train_x=train_z_umap,
                train_y=train_y,
                n_iter=sdrgp_iters,
                lr=sdrgp_lr,
            )
            seed_runtimes["supervised_umap_gp_train"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            supervised_umap_gp_train_metrics = _gp_split_metrics(
                supervised_umap_gp_model,
                supervised_umap_gp_likelihood,
                train_z_umap,
                train_y,
            )
            supervised_umap_gp_test_metrics = _gp_split_metrics(
                supervised_umap_gp_model,
                supervised_umap_gp_likelihood,
                test_z_umap,
                test_y,
            )
            seed_runtimes["supervised_umap_gp_eval"] = time.perf_counter() - t0

            seed_metrics["supervised_umap_gp"] = {
                "train": supervised_umap_gp_train_metrics,
                "test": supervised_umap_gp_test_metrics,
            }
            seed_logs["supervised_umap_gp"] = supervised_umap_gp_logs

        if "sdrgp" in selected_methods:
            sdr_kwargs = dict(base_sdr_kwargs)
            sdr_kwargs["output_sam"] = train_x.shape[-2]
            sdr_kwargs["seed"] = int(seed)
            dr = SDR(**sdr_kwargs)

            t0 = time.perf_counter()
            Z_train, _ = dr.fit_transform(train_x, train_y)
            Z_test = dr.transform(test_x)
            seed_runtimes["sdr_fit"] = time.perf_counter() - t0

            train_z, test_z = _normalize_with_train_stats(Z_train, Z_test)

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
            seed_runtimes["sdrgp_train"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            sdrgp_train_metrics = _gp_split_metrics(sdrgp_model, sdrgp_likelihood, train_z, train_y)
            sdrgp_test_metrics = _gp_split_metrics(sdrgp_model, sdrgp_likelihood, test_z, test_y)
            seed_runtimes["sdrgp_eval"] = time.perf_counter() - t0

            seed_metrics["sdrgp"] = {"train": sdrgp_train_metrics, "test": sdrgp_test_metrics}
            seed_logs["sdrgp"] = sdrgp_logs

        seed_runtimes["total"] = float(sum(seed_runtimes.values()))

        seed_result = {
            "dataset": dataset_name,
            "seed": int(seed),
            "methods": list(selected_methods),
            "runtimes_s": seed_runtimes,
            "metrics": seed_metrics,
            "logs": seed_logs,
        }
        per_seed_results.append(seed_result)

        for model_name, split_metrics in seed_result["metrics"].items():
            metric_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "model": model_name,
                    "train_mse": split_metrics["train"]["mse"],
                    "test_mse": split_metrics["test"]["mse"],
                    "train_mean_log_likelihood": split_metrics["train"]["mean_log_likelihood"],
                    "test_mean_log_likelihood": split_metrics["test"]["mean_log_likelihood"],
                }
            )

        runtime_rows.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                **seed_result["runtimes_s"],
            }
        )

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
        description="Run GP regression benchmarks from gp_benchmarks.py."
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
        "--methods",
        type=str,
        nargs="+",
        default=list(SUPPORTED_METHODS),
        choices=SUPPORTED_METHODS,
        help="Subset of methods to run.",
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
    parser.add_argument("--dkl-iters", type=int, default=20)
    parser.add_argument("--sdrgp-iters", type=int, default=20)
    parser.add_argument("--dkl-lr", type=float, default=0.1)
    parser.add_argument("--sdrgp-lr", type=float, default=0.1)
    parser.add_argument("--dkl-embedding-dim", type=int, default=2)
    parser.add_argument("--sdr-output-dim", type=int, default=2)
    parser.add_argument("--nca-max-iter", type=int, default=100)
    parser.add_argument("--nca-tol", type=float, default=1e-5)
    parser.add_argument("--nca-num-bins", type=int, default=10)
    parser.add_argument("--deepgp-iters", type=int, default=20)
    parser.add_argument("--deepgp-lr", type=float, default=0.1)
    parser.add_argument("--deepgp-hidden-dim", type=int, default=2)
    parser.add_argument("--deepgp-num-inducing", type=int, default=128)
    parser.add_argument("--deepgp-likelihood-samples", type=int, default=16)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_outputs/gp_regression_benchmarks",
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
        result = benchmark_dkl_vs_sdrgp_dataset(
            dataset_name=dataset,
            seeds=args.seeds,
            methods=args.methods,
            data_path=data_path,
            test_size=args.test_size,
            dkl_iters=args.dkl_iters,
            sdrgp_iters=args.sdrgp_iters,
            dkl_lr=args.dkl_lr,
            sdrgp_lr=args.sdrgp_lr,
            dkl_embedding_dim=args.dkl_embedding_dim,
            sdr_output_dim=args.sdr_output_dim,
            nca_max_iter=args.nca_max_iter,
            nca_tol=args.nca_tol,
            nca_num_bins=args.nca_num_bins,
            deepgp_iters=args.deepgp_iters,
            deepgp_lr=args.deepgp_lr,
            deepgp_hidden_dim=args.deepgp_hidden_dim,
            deepgp_num_inducing=args.deepgp_num_inducing,
            deepgp_likelihood_samples=args.deepgp_likelihood_samples,
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
            "methods": list(_normalize_requested_methods(args.methods)),
            "per_seed_results": result["per_seed_results"],
            "metrics": result["metrics_df"].to_dict(orient="records"),
            "metrics_summary_by_model": {
                model_name: summary_df.to_dict(orient="records")
                for model_name, summary_df in result["metrics_summary_by_model"].items()
            },
            "runtimes": result["runtime_df"].to_dict(orient="records"),
        }
        full_json_path.write_text(json.dumps(serializable_result, indent=2))

        print(f"[{dataset}] Methods: {', '.join(_normalize_requested_methods(args.methods))}")
        print(f"[{dataset}] Wrote: {metrics_path}")
        for model_name, summary_path in summary_paths.items():
            print(f"[{dataset}] Wrote: {summary_path}")
        print(f"[{dataset}] Wrote: {runtime_path}")
        print(f"[{dataset}] Wrote: {full_json_path}")
        print(f"\n[{dataset}] Metrics preview:")
        print(result["metrics_df"].to_string(index=False))


if __name__ == "__main__":
    main()
