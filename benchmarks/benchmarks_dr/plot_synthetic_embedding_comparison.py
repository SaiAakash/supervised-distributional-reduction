from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPL_CONFIG_DIR = Path(".matplotlib")
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from sklearn.datasets import make_s_curve, make_swiss_roll
from sklearn.model_selection import train_test_split

import umap
from src.affinities import NormalizedGaussianAndStudentAffinity, SymmetricEntropicAffinity
from src.kernel_supervised_pca import KernelSupervisedPCA
from src.sdr import SDR


DATASETS = [
    ("s_curve", "S-curve"),
    ("swiss_roll", "Swiss roll"),
]

COLUMN_TITLES = ["Original data", "KSPCA", "S-UMAP", "SDR"]

TITLE_FONT_SIZE = 40
AXIS_LABEL_FONT_SIZE = 35
TICK_FONT_SIZE = 30
COLORBAR_FONT_SIZE = 30

POINT_SIZE = 34
CMAP = "viridis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 2x4 figure comparing original synthetic datasets against "
            "KSPCA, supervised UMAP, and SDR embeddings."
        )
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of samples to generate per dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for dataset generation and reducers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs/synthetic_embedding_plots"),
        help="Directory where the combined figure will be saved.",
    )
    parser.add_argument(
        "--output-name",
        default="pure_dr.pdf",
        help="Filename for the combined figure.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to use for SDR and KSPCA.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of inner SDR optimization iterations.",
    )
    parser.add_argument(
        "--max-iter-outer",
        type=int,
        default=20,
        help="Maximum number of outer SDR optimization iterations.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved figure DPI.",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_dataset(dataset_name: str, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if dataset_name == "s_curve":
        x, y = make_s_curve(
            n_samples=n_samples,
            noise=0.0,
            random_state=seed,
        )
        return x.astype(np.float64), y.astype(np.float64)
    if dataset_name == "swiss_roll":
        x, y = make_swiss_roll(
            n_samples=n_samples,
            noise=0.0,
            random_state=seed,
        )
        return x.astype(np.float64), y.astype(np.float64)
    raise ValueError(f"Unsupported dataset {dataset_name!r}")


def _normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True)
    return (x - mean) / std


def _prepare_dataset(
    dataset_name: str,
    n_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = _make_dataset(dataset_name=dataset_name, n_samples=n_samples, seed=seed)
    train_x_np, _, train_y_np, _ = train_test_split(
        x,
        y,
        shuffle=True,
        random_state=seed,
    )

    train_x = _normalize_tensor(torch.tensor(train_x_np, dtype=torch.double))
    train_y = _normalize_tensor(torch.tensor(train_y_np, dtype=torch.double)).unsqueeze(-1)
    return train_x, train_y


def _fit_kspca_embedding(
    x: torch.Tensor,
    y: torch.Tensor,
) -> np.ndarray:
    reducer = KernelSupervisedPCA(n_components=2)
    embedding = reducer.fit_transform(x, y)
    return embedding.detach().cpu().numpy()


def _fit_umap_embedding(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=50,
        min_dist=0.1,
        n_components=2,
        random_state=seed,
    )
    return reducer.fit_transform(
        x.detach().cpu().numpy(),
        y=y.detach().cpu().numpy().ravel(),
    )


def _fit_sdr_embedding(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
) -> np.ndarray:
    reducer = SDR(
        affinity_data=SymmetricEntropicAffinity(perp=50, lr=0.01, verbose=False),
        affinity_embedding=NormalizedGaussianAndStudentAffinity(student=True),
        hsic_kernel_z="rq",
        hsic_kernel_y="rbf",
        fgw_loss="square_loss",
        loss_fun="kl_loss",
        alpha=0.2,
        output_sam=x.shape[0],
        output_dim=2,
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
    x_device = x.to(device)
    y_device = y.to(device)
    embedding, _ = reducer.fit_transform(x_device, y_device)
    return embedding.detach().cpu().numpy()


def _plot_original_axis(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    norm: colors.Normalize,
) -> None:
    ax.scatter(
        x[:, 0],
        x[:, 1],
        x[:, 2],
        c=y,
        cmap=CMAP,
        norm=norm,
        s=POINT_SIZE,
        alpha=0.9,
    )
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE, pad=8)
    ax.zaxis.set_tick_params(labelsize=TICK_FONT_SIZE, pad=10)
    ax.view_init(elev=15.0, azim=-75, roll=0)
    ax.set_box_aspect((1, 1, 1))


def _plot_embedding_axis(
    ax: plt.Axes,
    embedding: np.ndarray,
    y: np.ndarray,
    norm: colors.Normalize,
) -> None:
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=y,
        cmap=CMAP,
        norm=norm,
        s=POINT_SIZE,
        alpha=0.9,
    )
    ax.tick_params(axis="both", which="both", labelsize=TICK_FONT_SIZE)
    ax.set_box_aspect(1)


def plot_embedding_grid(
    output_path: Path,
    n_samples: int,
    seed: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(30, 18))
    grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.48, hspace=0.10)

    row_axes: list[list[plt.Axes]] = []
    shared_scatter = None
    for row_idx, (dataset_name, dataset_title) in enumerate(DATASETS):
        x_tensor, y_tensor = _prepare_dataset(
            dataset_name=dataset_name,
            n_samples=n_samples,
            seed=seed,
        )
        plot_x = x_tensor.detach().cpu().numpy()
        plot_y = y_tensor.detach().cpu().numpy().ravel()

        kspca_embedding = _fit_kspca_embedding(x=x_tensor, y=y_tensor)
        umap_embedding = _fit_umap_embedding(x=x_tensor, y=y_tensor, seed=seed)
        sdr_embedding = _fit_sdr_embedding(
            x=x_tensor,
            y=y_tensor,
            seed=seed,
            device=device,
            max_iter=max_iter,
            max_iter_outer=max_iter_outer,
        )

        norm = colors.Normalize(vmin=float(plot_y.min()), vmax=float(plot_y.max()))
        current_row_axes: list[plt.Axes] = []

        original_ax = fig.add_subplot(grid[row_idx, 0], projection="3d")
        _plot_original_axis(original_ax, plot_x, plot_y, norm)
        original_ax.text2D(
            -0.28,
            0.5,
            dataset_title,
            transform=original_ax.transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=TITLE_FONT_SIZE,
        )
        current_row_axes.append(original_ax)

        kspca_ax = fig.add_subplot(grid[row_idx, 1])
        _plot_embedding_axis(kspca_ax, kspca_embedding, plot_y, norm)
        current_row_axes.append(kspca_ax)

        umap_ax = fig.add_subplot(grid[row_idx, 2])
        _plot_embedding_axis(umap_ax, umap_embedding, plot_y, norm)
        current_row_axes.append(umap_ax)

        sdr_ax = fig.add_subplot(grid[row_idx, 3])
        _plot_embedding_axis(sdr_ax, sdr_embedding, plot_y, norm)
        current_row_axes.append(sdr_ax)
        shared_scatter = sdr_ax.collections[0]

        row_axes.append(current_row_axes)

    for col_idx, title in enumerate(COLUMN_TITLES):
        row_axes[0][col_idx].set_title(title, fontsize=TITLE_FONT_SIZE, pad=20)

    if shared_scatter is not None:
        colorbar_ax = fig.add_subplot(grid[:, 4])
        colorbar = fig.colorbar(shared_scatter, cax=colorbar_ax)
        colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
        colorbar.set_label("Target Value", fontsize=COLORBAR_FONT_SIZE)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    torch.set_default_dtype(torch.double)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name

    plot_embedding_grid(
        output_path=output_path,
        n_samples=args.n_samples,
        seed=args.seed,
        device=args.device,
        max_iter=args.max_iter,
        max_iter_outer=args.max_iter_outer,
        dpi=args.dpi,
    )
    print(f"Saved synthetic embedding comparison grid to {output_path}")


if __name__ == "__main__":
    main()
