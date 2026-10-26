import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gpytorch
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NeighborhoodComponentsAnalysis

from src.affinities import NormalizedGaussianAndStudentAffinity, SymmetricEntropicAffinity
from src.kernel_supervised_pca import KernelSupervisedPCA
from src.sdr import SDR


SUPPORTED_IMAGE_DATASETS = ("coil20", "mnist")
SUPPORTED_IMAGE_METHODS = (
    "dkl",
    "kernel_supervised_pca_gp",
    "nca_gp",
    "supervised_umap_gp",
    "sdrgp",
)


def _normalize_requested_methods(
    methods: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if methods is None:
        return SUPPORTED_IMAGE_METHODS
    normalized = tuple(str(method).lower() for method in methods)
    invalid = sorted(set(normalized) - set(SUPPORTED_IMAGE_METHODS))
    if invalid:
        raise ValueError(
            f"Unknown methods {invalid}. Supported methods: {list(SUPPORTED_IMAGE_METHODS)}"
        )
    return tuple(dict.fromkeys(normalized))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_with_train_stats(
    train_t: torch.Tensor, test_t: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    dims = (0,)
    mu = train_t.mean(dim=dims, keepdim=True)
    std = train_t.std(dim=dims, keepdim=True).clamp_min(eps)
    return (train_t - mu) / std, (test_t - mu) / std


def _ensure_grayscale_image_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x
    if x.ndim != 2:
        raise ValueError(f"Expected flattened images or NCHW tensors, got shape {tuple(x.shape)}")

    num_features = x.shape[-1]
    side = int(round(num_features**0.5))
    if side * side != num_features:
        raise ValueError(
            f"Cannot reshape flattened features of size {num_features} into square images."
        )
    return x.view(x.shape[0], 1, side, side)


def _load_coil20(data_path: str | Path | None) -> tuple[np.ndarray, np.ndarray]:
    candidates: list[Path] = []
    if data_path is not None:
        candidates.append(Path(data_path))
    candidates.extend(
        [
            Path("data/coil20.pt"),
            Path(__file__).resolve().parent.parent / "data" / "coil20.pt",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        data = torch.load(path, map_location="cpu")
        x = data["X"]
        y = data["y"]
        return x.detach().cpu().numpy(), y.detach().cpu().numpy()
    raise FileNotFoundError(
        "Could not locate COIL20 data. Expected a torch file like data/coil20.pt containing "
        "{'X': ..., 'y': ...}."
    )


def _load_mnist(data_path: str | Path | None) -> tuple[np.ndarray, np.ndarray]:
    candidates: list[Path] = []
    if data_path is not None:
        candidates.append(Path(data_path))
    candidates.extend(
        [
            Path("data/mnist_test_stratified_2k.pt"),
            Path(__file__).resolve().parent.parent / "data" / "mnist_test_stratified_2k.pt",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".npz":
            data = np.load(path)
            return data["X"], data["y"]
        if path.suffix in {".pt", ".pth"}:
            data = torch.load(path, map_location="cpu")
            x = data["X"]
            y = data["y"]
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu().numpy()
            return np.asarray(x), np.asarray(y)
        raise ValueError("Unsupported MNIST file format. Use .npz or .pt/.pth.")

    from torchvision import datasets

    train = datasets.MNIST(root=str(Path("data/torchvision")), train=True, download=True)
    test = datasets.MNIST(root=str(Path("data/torchvision")), train=False, download=True)
    x = torch.cat([train.data, test.data], dim=0).view(-1, 28 * 28).cpu().numpy()
    y = torch.cat([train.targets, test.targets], dim=0).cpu().numpy()
    return x, y


def _load_image_dataset_xy(
    dataset_name: str, data_path: str | Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    dataset_name = dataset_name.lower()
    if dataset_name == "coil20":
        return _load_coil20(data_path)
    if dataset_name == "mnist":
        return _load_mnist(data_path)
    raise ValueError(
        f"Unknown dataset '{dataset_name}'. Supported datasets: {SUPPORTED_IMAGE_DATASETS}"
    )


class _PaperCNNFeatureExtractor(torch.nn.Module):
    """
    CNN encoder following the feature extractor used in the DKL paper:
    conv(20, 5x5) -> pool -> conv(50, 5x5) -> pool -> fc(1000) -> fc(500) -> fc(50) -> fc(k).
    """

    def __init__(self, image_shape: tuple[int, int], embedding_dim: int):
        super().__init__()
        self.image_shape = image_shape
        self.conv1 = torch.nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(20, 50, kernel_size=5)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        flattened_dim = self._infer_flattened_dim(image_shape)
        self.fc1 = torch.nn.Linear(flattened_dim, 1000)
        self.fc2 = torch.nn.Linear(1000, 500)
        self.fc3 = torch.nn.Linear(500, 50)
        self.fc4 = torch.nn.Linear(50, embedding_dim)

    def _infer_flattened_dim(self, image_shape: tuple[int, int]) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, image_shape[0], image_shape[1])
            dummy = self.pool(torch.relu(self.conv1(dummy)))
            dummy = self.pool(torch.relu(self.conv2(dummy)))
        return int(dummy.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(x.shape[0], 1, self.image_shape[0], self.image_shape[1])
        elif x.ndim != 4:
            raise ValueError(f"Expected flattened or NCHW image tensor, got shape {tuple(x.shape)}")
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.fc4(x)


class _DirichletDKLModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.DirichletClassificationLikelihood,
        image_shape: tuple[int, int],
        embedding_dim: int,
    ):
        super().__init__(train_x, train_y, likelihood)
        batch_shape = torch.Size((likelihood.num_classes,))
        self.feature_extractor = _PaperCNNFeatureExtractor(image_shape, embedding_dim)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=embedding_dim),
            batch_shape=batch_shape,
        )
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        projected_x = self.scale_to_bounds(self.feature_extractor(x))
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class _DirichletExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.DirichletClassificationLikelihood,
        ard_num_dims: int,
    ):
        super().__init__(train_x, train_y, likelihood)
        batch_shape = torch.Size((likelihood.num_classes,))
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=ard_num_dims),
            batch_shape=batch_shape,
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _kernel_params(model: gpytorch.models.ExactGP) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        out["outputscale_mean"] = float(model.covar_module.outputscale.detach().mean().item())
    except Exception:
        pass
    try:
        out["lengthscale_mean"] = float(
            model.covar_module.base_kernel.lengthscale.detach().mean().item()
        )
    except Exception:
        pass
    try:
        out["noise_mean"] = float(model.likelihood.noise.detach().mean().item())
    except Exception:
        pass
    try:
        out["second_noise_mean"] = float(
            model.likelihood.second_noise_covar.noise.detach().mean().item()
        )
    except Exception:
        pass
    return out


def _train_dirichlet_gp(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.DirichletClassificationLikelihood,
    train_x: torch.Tensor,
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
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        neg_mll = -mll(output, likelihood.transformed_targets).sum()
        neg_mll.backward()
        optimizer.step()

        step = {
            "iter": float(i + 1),
            "neg_mll": float(neg_mll.item()),
            "mll": float((-neg_mll).item()),
        }
        step.update(_kernel_params(model))
        logs.append(step)
    return logs


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
    train_z_np = reducer.fit_transform(
        train_x.detach().cpu().numpy(),
        y=train_y.detach().cpu().numpy().ravel(),
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


def _fit_nca_embeddings(
    train_x: torch.Tensor,
    test_x: torch.Tensor,
    train_y: torch.Tensor,
    output_dim: int,
    seed: int,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    nca = NeighborhoodComponentsAnalysis(
        n_components=output_dim,
        init="pca",
        random_state=seed,
        max_iter=max_iter,
        tol=tol,
    )
    train_z_np = nca.fit_transform(
        train_x.detach().cpu().numpy(),
        train_y.detach().cpu().numpy().ravel(),
    )
    test_z_np = nca.transform(test_x.detach().cpu().numpy())
    train_z = torch.as_tensor(train_z_np, device=train_x.device, dtype=train_x.dtype)
    test_z = torch.as_tensor(test_z_np, device=test_x.device, dtype=test_x.dtype)
    return train_z, test_z


def _predict_class_probabilities(
    model: gpytorch.models.ExactGP,
    x: torch.Tensor,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent_dist = model(x)
        logits = latent_dist.mean.transpose(0, 1)
        latent_samples = latent_dist.sample(torch.Size((num_samples,))).exp()
        probabilities = (
            (latent_samples / latent_samples.sum(dim=-2, keepdim=True).clamp_min(1e-12))
            .mean(dim=0)
            .transpose(0, 1)
        )
    return logits, probabilities


def _classification_metrics(
    model: gpytorch.models.ExactGP,
    x: torch.Tensor,
    y: torch.Tensor,
    prob_samples: int,
) -> dict[str, float]:
    logits, probabilities = _predict_class_probabilities(model, x, num_samples=prob_samples)
    predictions = probabilities.argmax(dim=-1)
    accuracy = (predictions == y).double().mean()
    mean_log_prob = torch.log(probabilities.gather(-1, y.unsqueeze(-1)).clamp_min(1e-12)).mean()
    return {
        "accuracy": float(accuracy.item()),
        "mean_log_probability": float(mean_log_prob.item()),
        "mean_max_probability": float(probabilities.max(dim=-1).values.mean().item()),
        "logit_magnitude_mean": float(logits.abs().mean().item()),
    }


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


def benchmark_dkl_vs_sdrgp_image_classification(
    dataset_name: str,
    seeds: list[int] | tuple[int, ...],
    methods: list[str] | tuple[str, ...] | None = None,
    data_path: str | Path | None = None,
    test_size: float = 0.2,
    dkl_iters: int = 50,
    sdrgp_iters: int = 50,
    dkl_lr: float = 0.1,
    sdrgp_lr: float = 0.1,
    dkl_embedding_dim: int = 10,
    sdr_output_dim: int = 10,
    supervised_umap_n_neighbors: int = 50,
    supervised_umap_min_dist: float = 0.1,
    supervised_umap_target_metric: str = "l2",
    kernel_supervised_pca_kernel: str = "rbf",
    kernel_supervised_pca_gamma: float | None = None,
    kernel_supervised_pca_target_kernel: str = "rbf",
    kernel_supervised_pca_target_gamma: float | None = None,
    kernel_supervised_pca_reg: float = 1e-6,
    nca_max_iter: int = 100,
    nca_tol: float = 1e-5,
    alpha_epsilon: float = 0.01,
    likelihood_samples: int = 256,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.double,
    sdr_overrides: dict[str, Any] | None = None,
    preprocess: bool = True,
) -> dict[str, Any]:
    dataset_name = dataset_name.lower()
    selected_methods = _normalize_requested_methods(methods)
    x_np, y_np = _load_image_dataset_xy(dataset_name, data_path=data_path)
    device = torch.device(device)

    base_sdr_kwargs: dict[str, Any] = {
        "affinity_data": SymmetricEntropicAffinity(perp=20, lr=1),
        "affinity_embedding": NormalizedGaussianAndStudentAffinity(student=True),
        "hsic_kernel_z": "rq",
        "hsic_kernel_y": "delta",
        "fgw_loss": "ce_loss",
        "loss_fun": "kl_loss",
        "alpha": 0.2,
        "output_dim": sdr_output_dim,
        "max_iter_outer": 20,
        "tol": 1e-5,
        "lambda_hsic": 1000.0,
        "init_T": "random",
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
        print(f"Running seed {seed}...")
        _set_seed(int(seed))
        x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(
            x_np,
            y_np,
            test_size=test_size,
            random_state=int(seed),
            stratify=y_np,
        )

        train_x_flat = torch.tensor(x_train_np, dtype=dtype, device=device)
        test_x_flat = torch.tensor(x_test_np, dtype=dtype, device=device)
        train_y = torch.tensor(y_train_np, dtype=torch.long, device=device)
        test_y = torch.tensor(y_test_np, dtype=torch.long, device=device)

        train_x_img = _ensure_grayscale_image_tensor(train_x_flat)
        image_shape = (train_x_img.shape[-2], train_x_img.shape[-1])
        if dataset_name == "mnist":
            train_x_flat_methods, test_x_flat_methods = train_x_flat, test_x_flat
        else:
            train_x_flat_methods, test_x_flat_methods = _normalize_with_train_stats(
                train_x_flat, test_x_flat
            )
        train_x_flat_sdr = train_x_flat.clone()
        test_x_flat_sdr = test_x_flat.clone()
        if preprocess and dataset_name != "mnist":
            train_x_flat_sdr, test_x_flat_sdr = _normalize_with_train_stats(
                train_x_flat_sdr, test_x_flat_sdr
            )

        seed_metrics: dict[str, dict[str, dict[str, float]]] = {}
        seed_logs: dict[str, Any] = {}
        seed_runtimes: dict[str, float] = {}
        num_classes = int(torch.unique(train_y).numel())

        if "dkl" in selected_methods:
            dkl_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
                targets=train_y,
                alpha_epsilon=alpha_epsilon,
                learn_additional_noise=True,
                dtype=dtype,
            ).to(device=device, dtype=dtype)
            dkl_model = _DirichletDKLModel(
                train_x=train_x_flat_methods,
                train_y=dkl_likelihood.transformed_targets,
                likelihood=dkl_likelihood,
                image_shape=image_shape,
                embedding_dim=dkl_embedding_dim,
            ).to(device=device, dtype=dtype)
            dkl_param_groups = [
                {"params": dkl_model.feature_extractor.parameters()},
                {"params": dkl_model.covar_module.parameters()},
                {"params": dkl_model.mean_module.parameters()},
                {"params": dkl_model.likelihood.parameters()},
            ]

            t0 = time.perf_counter()
            dkl_logs = _train_dirichlet_gp(
                model=dkl_model,
                likelihood=dkl_likelihood,
                train_x=train_x_flat_methods,
                n_iter=dkl_iters,
                lr=dkl_lr,
                parameter_groups=dkl_param_groups,
            )
            seed_runtimes["dkl_train"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            dkl_train_metrics = _classification_metrics(
                dkl_model, train_x_flat_methods, train_y, prob_samples=likelihood_samples
            )
            dkl_test_metrics = _classification_metrics(
                dkl_model, test_x_flat_methods, test_y, prob_samples=likelihood_samples
            )
            seed_runtimes["dkl_eval"] = time.perf_counter() - t0

            seed_metrics["dkl"] = {"train": dkl_train_metrics, "test": dkl_test_metrics}
            seed_logs["dkl"] = dkl_logs
            num_classes = int(dkl_likelihood.num_classes)

        if "kernel_supervised_pca_gp" in selected_methods:
            t0 = time.perf_counter()
            train_z_kspca, test_z_kspca = _fit_kernel_supervised_pca_embeddings(
                train_x=train_x_flat_methods,
                test_x=test_x_flat_methods,
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

            kspca_gp_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
                targets=train_y,
                alpha_epsilon=alpha_epsilon,
                learn_additional_noise=True,
                dtype=dtype,
            ).to(device=device, dtype=dtype)
            kspca_gp_model = _DirichletExactGPModel(
                train_x=train_z_kspca,
                train_y=kspca_gp_likelihood.transformed_targets,
                likelihood=kspca_gp_likelihood,
                ard_num_dims=sdr_output_dim,
            ).to(device=device, dtype=dtype)

            t0 = time.perf_counter()
            kspca_gp_logs = _train_dirichlet_gp(
                model=kspca_gp_model,
                likelihood=kspca_gp_likelihood,
                train_x=train_z_kspca,
                n_iter=sdrgp_iters,
                lr=sdrgp_lr,
            )
            seed_runtimes["kernel_supervised_pca_gp_train"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            kspca_gp_train_metrics = _classification_metrics(
                kspca_gp_model, train_z_kspca, train_y, prob_samples=likelihood_samples
            )
            kspca_gp_test_metrics = _classification_metrics(
                kspca_gp_model, test_z_kspca, test_y, prob_samples=likelihood_samples
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
                train_x=train_x_flat_methods,
                test_x=test_x_flat_methods,
                train_y=train_y,
                output_dim=sdr_output_dim,
                seed=int(seed),
                max_iter=nca_max_iter,
                tol=nca_tol,
            )
            seed_runtimes["nca_fit"] = time.perf_counter() - t0

            train_z_nca, test_z_nca = _normalize_with_train_stats(train_z_nca, test_z_nca)

            nca_gp_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
                targets=train_y,
                alpha_epsilon=alpha_epsilon,
                learn_additional_noise=True,
                dtype=dtype,
            ).to(device=device, dtype=dtype)
            nca_gp_model = _DirichletExactGPModel(
                train_x=train_z_nca,
                train_y=nca_gp_likelihood.transformed_targets,
                likelihood=nca_gp_likelihood,
                ard_num_dims=sdr_output_dim,
            ).to(device=device, dtype=dtype)

            t0 = time.perf_counter()
            nca_gp_logs = _train_dirichlet_gp(
                model=nca_gp_model,
                likelihood=nca_gp_likelihood,
                train_x=train_z_nca,
                n_iter=sdrgp_iters,
                lr=sdrgp_lr,
            )
            seed_runtimes["nca_gp_train"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            nca_gp_train_metrics = _classification_metrics(
                nca_gp_model, train_z_nca, train_y, prob_samples=likelihood_samples
            )
            nca_gp_test_metrics = _classification_metrics(
                nca_gp_model, test_z_nca, test_y, prob_samples=likelihood_samples
            )
            seed_runtimes["nca_gp_eval"] = time.perf_counter() - t0

            seed_metrics["nca_gp"] = {"train": nca_gp_train_metrics, "test": nca_gp_test_metrics}
            seed_logs["nca_gp"] = nca_gp_logs

        if "supervised_umap_gp" in selected_methods:
            t0 = time.perf_counter()
            train_z_umap, test_z_umap = _fit_supervised_umap_embeddings(
                train_x=train_x_flat_methods,
                test_x=test_x_flat_methods,
                train_y=train_y,
                output_dim=sdr_output_dim,
                seed=int(seed),
                n_neighbors=supervised_umap_n_neighbors,
                min_dist=supervised_umap_min_dist,
                target_metric=supervised_umap_target_metric,
            )
            seed_runtimes["supervised_umap_fit"] = time.perf_counter() - t0

            train_z_umap, test_z_umap = _normalize_with_train_stats(train_z_umap, test_z_umap)

            supervised_umap_gp_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
                targets=train_y,
                alpha_epsilon=alpha_epsilon,
                learn_additional_noise=True,
                dtype=dtype,
            ).to(device=device, dtype=dtype)
            supervised_umap_gp_model = _DirichletExactGPModel(
                train_x=train_z_umap,
                train_y=supervised_umap_gp_likelihood.transformed_targets,
                likelihood=supervised_umap_gp_likelihood,
                ard_num_dims=sdr_output_dim,
            ).to(device=device, dtype=dtype)

            t0 = time.perf_counter()
            supervised_umap_gp_logs = _train_dirichlet_gp(
                model=supervised_umap_gp_model,
                likelihood=supervised_umap_gp_likelihood,
                train_x=train_z_umap,
                n_iter=sdrgp_iters,
                lr=sdrgp_lr,
            )
            seed_runtimes["supervised_umap_gp_train"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            supervised_umap_gp_train_metrics = _classification_metrics(
                supervised_umap_gp_model, train_z_umap, train_y, prob_samples=likelihood_samples
            )
            supervised_umap_gp_test_metrics = _classification_metrics(
                supervised_umap_gp_model, test_z_umap, test_y, prob_samples=likelihood_samples
            )
            seed_runtimes["supervised_umap_gp_eval"] = time.perf_counter() - t0

            seed_metrics["supervised_umap_gp"] = {
                "train": supervised_umap_gp_train_metrics,
                "test": supervised_umap_gp_test_metrics,
            }
            seed_logs["supervised_umap_gp"] = supervised_umap_gp_logs

        if "sdrgp" in selected_methods:
            sdr_kwargs = dict(base_sdr_kwargs)
            sdr_kwargs["output_sam"] = train_x_flat_sdr.shape[-2]
            sdr_kwargs["seed"] = int(seed)
            dr = SDR(**sdr_kwargs)

            t0 = time.perf_counter()
            train_z, _ = dr.fit_transform(train_x_flat_sdr, train_y)
            test_z = dr.transform(test_x_flat_sdr)
            seed_runtimes["sdr_fit"] = time.perf_counter() - t0

            train_z, test_z = _normalize_with_train_stats(train_z, test_z)

            sdrgp_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
                targets=train_y,
                alpha_epsilon=alpha_epsilon,
                learn_additional_noise=True,
                dtype=dtype,
            ).to(device=device, dtype=dtype)
            sdrgp_model = _DirichletExactGPModel(
                train_x=train_z,
                train_y=sdrgp_likelihood.transformed_targets,
                likelihood=sdrgp_likelihood,
                ard_num_dims=sdr_output_dim,
            ).to(device=device, dtype=dtype)

            t0 = time.perf_counter()
            sdrgp_logs = _train_dirichlet_gp(
                model=sdrgp_model,
                likelihood=sdrgp_likelihood,
                train_x=train_z,
                n_iter=sdrgp_iters,
                lr=sdrgp_lr,
            )
            seed_runtimes["sdrgp_train"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            sdrgp_train_metrics = _classification_metrics(
                sdrgp_model, train_z, train_y, prob_samples=likelihood_samples
            )
            sdrgp_test_metrics = _classification_metrics(
                sdrgp_model, test_z, test_y, prob_samples=likelihood_samples
            )
            seed_runtimes["sdrgp_eval"] = time.perf_counter() - t0

            seed_metrics["sdrgp"] = {"train": sdrgp_train_metrics, "test": sdrgp_test_metrics}
            seed_logs["sdrgp"] = sdrgp_logs

        seed_runtimes["total"] = float(sum(seed_runtimes.values()))

        seed_result = {
            "dataset": dataset_name,
            "seed": int(seed),
            "methods": list(selected_methods),
            "num_classes": num_classes,
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
                    "train_accuracy": split_metrics["train"]["accuracy"],
                    "test_accuracy": split_metrics["test"]["accuracy"],
                    "train_mean_log_probability": split_metrics["train"]["mean_log_probability"],
                    "test_mean_log_probability": split_metrics["test"]["mean_log_probability"],
                    "train_mean_max_probability": split_metrics["train"]["mean_max_probability"],
                    "test_mean_max_probability": split_metrics["test"]["mean_max_probability"],
                    "train_logit_magnitude_mean": split_metrics["train"]["logit_magnitude_mean"],
                    "test_logit_magnitude_mean": split_metrics["test"]["logit_magnitude_mean"],
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
    return {
        "dataset": dataset_name,
        "per_seed_results": per_seed_results,
        "metrics_df": metrics_df,
        "metrics_summary_df": _summarize_metrics_across_seeds(metrics_df),
        "runtime_df": pd.DataFrame(runtime_rows),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DKL vs SDRGP on image classification datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(SUPPORTED_IMAGE_DATASETS),
        choices=SUPPORTED_IMAGE_DATASETS,
        help="Image datasets to benchmark.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(SUPPORTED_IMAGE_METHODS),
        choices=SUPPORTED_IMAGE_METHODS,
        help="Subset of methods to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds to evaluate.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Held-out test split fraction.",
    )
    parser.add_argument("--dkl-iters", type=int, default=50)
    parser.add_argument("--sdrgp-iters", type=int, default=50)
    parser.add_argument("--dkl-lr", type=float, default=0.1)
    parser.add_argument("--sdrgp-lr", type=float, default=0.1)
    parser.add_argument("--dkl-embedding-dim", type=int, default=10)
    parser.add_argument("--sdr-output-dim", type=int, default=10)
    parser.add_argument("--alpha-epsilon", type=float, default=0.01)
    parser.add_argument("--likelihood-samples", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        default="benchmark_outputs/gp_classification_benchmarks",
        help="Directory where metrics/runtime CSVs and JSON summaries are written.",
    )
    parser.add_argument(
        "--coil20-path",
        default=None,
        help="Optional path to a local COIL20 .pt file.",
    )
    parser.add_argument(
        "--mnist-path",
        default=None,
        help="Optional path to a local MNIST .npz or .pt file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = {
        "coil20": args.coil20_path,
        "mnist": args.mnist_path,
    }

    for dataset_name in args.datasets:
        result = benchmark_dkl_vs_sdrgp_image_classification(
            dataset_name=dataset_name,
            seeds=args.seeds,
            methods=args.methods,
            data_path=dataset_paths[dataset_name],
            test_size=args.test_size,
            dkl_iters=args.dkl_iters,
            sdrgp_iters=args.sdrgp_iters,
            dkl_lr=args.dkl_lr,
            sdrgp_lr=args.sdrgp_lr,
            dkl_embedding_dim=args.dkl_embedding_dim,
            sdr_output_dim=args.sdr_output_dim,
            alpha_epsilon=args.alpha_epsilon,
            likelihood_samples=args.likelihood_samples,
            device=args.device,
        )
        print(f"[{dataset_name}] Methods: {', '.join(_normalize_requested_methods(args.methods))}")
        result["metrics_df"].to_csv(output_dir / f"{dataset_name}_metrics.csv", index=False)
        result["metrics_summary_df"].to_csv(
            output_dir / f"{dataset_name}_metrics_summary.csv", index=False
        )
        result["runtime_df"].to_csv(output_dir / f"{dataset_name}_runtimes.csv", index=False)
        with (output_dir / f"{dataset_name}_full_results.json").open("w") as fp:
            json.dump(result["per_seed_results"], fp, indent=2)


if __name__ == "__main__":
    main()
