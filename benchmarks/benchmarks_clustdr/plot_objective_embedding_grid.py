from __future__ import annotations

import argparse
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPL_CONFIG_DIR = Path(".matplotlib")
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.benchmarks_clustdr.classification_joint_benchmarks import _preprocess_features
from benchmarks.benchmarks_clustdr.sdr_objective_comparison import (
    BASE_SDR_KWARGS,
    DATASET_AFFINITY_KWARGS,
    DATASET_LOADERS,
)
from src.affinities import SymmetricEntropicAffinity
from src.sdr import SDR
from src.utils import plan_color


DATASET_ORDER = ["coil20", "snareseq"]
OBJECTIVE_ORDER = ["sdr", "fgw"]
DATASET_TITLES = {
    "coil20": "COIL-20",
    "snareseq": "SNAREseq",
}
OBJECTIVE_CAPTIONS = {
    "sdr": "SDR",
    "fgw": "FGW",
}

TITLE_FONT_SIZE = 40
AXIS_LABEL_FONT_SIZE = 35
TICK_FONT_SIZE = 30
COLORBAR_FONT_SIZE = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 1x4 grid of prototype embeddings for COIL-20 and SNAREseq "
            "under the SDR and FGW objectives."
        )
    )
    parser.add_argument(
        "--coil20-path",
        default=None,
        help="Optional path to the COIL-20 tensor file.",
    )
    parser.add_argument(
        "--snareseq-path",
        default=None,
        help="Optional path to the SNAREseq tensor file.",
    )
    parser.add_argument(
        "--output-sam",
        type=int,
        default=200,
        help="Number of prototypes in the learned embedding.",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=2,
        help="Embedding dimension.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="FGW interpolation weight used by SDR/FGW runs.",
    )
    parser.add_argument(
        "--lambda-hsic",
        type=float,
        default=1000.0,
        help="HSIC weight used by the SDR objective.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the embedding runs.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of inner optimization iterations.",
    )
    parser.add_argument(
        "--max-iter-outer",
        type=int,
        default=20,
        help="Maximum number of outer optimization iterations.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs/sdr_objective_comparison"),
        help="Directory where the combined figure will be written.",
    )
    parser.add_argument(
        "--output-name",
        default="objective_embedding_grid.pdf",
        help="Filename for the embedding figure.",
    )
    parser.add_argument(
        "--layout",
        choices=("1x4", "2x2"),
        default="1x4",
        help="Figure layout for the four panels. Defaults to 1x4.",
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


def _make_method(
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


def _run_embedding(
    dataset_name: str,
    objective: str,
    dataset_path: str | None,
    seed: int,
    output_sam: int,
    output_dim: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
    alpha: float,
    lambda_hsic: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _set_seed(seed)
    x, y = DATASET_LOADERS[dataset_name](dataset_path)
    x = _preprocess_features(x).to(device)
    y = y.to(device)

    if output_sam > x.shape[0]:
        raise ValueError(
            f"output_sam={output_sam} exceeds sample count {x.shape[0]} for {dataset_name}."
        )

    method = _make_method(
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
    z, _ = method.fit_transform(x, y)
    proto_labels, proto_sizes = plan_color(method.T.detach(), y.long())

    return (
        z.detach().cpu().numpy(),
        proto_labels.detach().cpu().numpy(),
        proto_sizes.detach().cpu().numpy(),
    )


def plot_objective_embedding_grid(
    output_path: Path,
    dataset_paths: dict[str, str | None],
    layout: str,
    seed: int,
    output_sam: int,
    output_dim: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
    alpha: float,
    lambda_hsic: float,
    dpi: int,
) -> None:
    if layout == "1x4":
        fig, axes = plt.subplots(1, 4, figsize=(26, 6.5))
        axes_by_panel = list(axes)
    elif layout == "2x2":
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes_by_panel = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    panel_idx = 0
    for dataset_name in DATASET_ORDER:
        for objective in OBJECTIVE_ORDER:
            ax = axes_by_panel[panel_idx]
            embedding, proto_labels, proto_sizes = _run_embedding(
                dataset_name=dataset_name,
                objective=objective,
                dataset_path=dataset_paths[dataset_name],
                seed=seed,
                output_sam=output_sam,
                output_dim=output_dim,
                device=device,
                max_iter=max_iter,
                max_iter_outer=max_iter_outer,
                alpha=alpha,
                lambda_hsic=lambda_hsic,
            )

            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=proto_labels,
                s=30 + 200 * (proto_sizes / np.maximum(proto_sizes.max(), 1e-12)),
                cmap="tab20",
                alpha=0.9,
                edgecolors="k",
                linewidths=0.3,
            )
            ax.tick_params(axis="both", which="both", labelsize=TICK_FONT_SIZE)
            ax.set_title(
                f"{DATASET_TITLES[dataset_name]} ({OBJECTIVE_CAPTIONS[objective]})",
                fontsize=TITLE_FONT_SIZE,
                pad=10,
            )
            panel_idx += 1

    if layout == "1x4":
        fig.tight_layout()
    else:
        fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.double)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    dataset_paths = {
        "coil20": args.coil20_path,
        "snareseq": args.snareseq_path,
    }

    plot_objective_embedding_grid(
        output_path=output_path,
        dataset_paths=dataset_paths,
        layout=args.layout,
        seed=args.seed,
        output_sam=args.output_sam,
        output_dim=args.output_dim,
        device=args.device,
        max_iter=args.max_iter,
        max_iter_outer=args.max_iter_outer,
        alpha=args.alpha,
        lambda_hsic=args.lambda_hsic,
        dpi=args.dpi,
    )
    print(f"Saved objective embedding grid to {output_path}")


if __name__ == "__main__":
    main()
