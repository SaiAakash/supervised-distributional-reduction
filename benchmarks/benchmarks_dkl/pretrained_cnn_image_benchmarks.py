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
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from benchmarks.benchmarks_dkl.image_classification_benchmarks import (
    SUPPORTED_IMAGE_DATASETS,
    _DirichletDKLModel,
    _PaperCNNFeatureExtractor,
    _classification_metrics,
    _ensure_grayscale_image_tensor,
    _load_image_dataset_xy,
    _normalize_with_train_stats,
    _set_seed,
    _summarize_metrics_across_seeds,
    _train_dirichlet_gp,
)


def _make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    effective_batch_size = min(int(batch_size), int(x.shape[0]))
    return DataLoader(
        TensorDataset(x, y),
        batch_size=max(1, effective_batch_size),
        shuffle=shuffle,
    )


class _PretrainCNNClassifier(torch.nn.Module):
    def __init__(
        self,
        image_shape: tuple[int, int],
        embedding_dim: int,
        num_classes: int,
        feature_extractor: _PaperCNNFeatureExtractor | None = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor or _PaperCNNFeatureExtractor(
            image_shape=image_shape,
            embedding_dim=embedding_dim,
        )
        self.classifier = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.feature_extractor(x)
        return self.classifier(embeddings)


def _classification_head_metrics(
    model: _PretrainCNNClassifier,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    logits_batches: list[torch.Tensor] = []
    labels_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for x_batch, y_batch in _make_loader(x, y, batch_size=batch_size, shuffle=False):
            logits_batches.append(model(x_batch))
            labels_batches.append(y_batch)
    if was_training:
        model.train()

    logits = torch.cat(logits_batches, dim=0)
    labels = torch.cat(labels_batches, dim=0)
    probabilities = torch.softmax(logits, dim=-1)
    predictions = probabilities.argmax(dim=-1)
    accuracy = (predictions == labels).double().mean()
    mean_log_prob = torch.log(
        probabilities.gather(-1, labels.unsqueeze(-1)).clamp_min(1e-12)
    ).mean()
    return {
        "accuracy": float(accuracy.item()),
        "mean_log_probability": float(mean_log_prob.item()),
        "mean_max_probability": float(probabilities.max(dim=-1).values.mean().item()),
        "logit_magnitude_mean": float(logits.abs().mean().item()),
    }


def _pretrain_feature_extractor(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    image_shape: tuple[int, int],
    embedding_dim: int,
    num_classes: int,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> tuple[_PaperCNNFeatureExtractor, list[dict[str, float]], dict[str, dict[str, float]]]:
    classifier = _PretrainCNNClassifier(
        image_shape=image_shape,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
    ).to(device=train_x.device, dtype=train_x.dtype)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)

    logs: list[dict[str, float]] = []
    for epoch in range(n_epochs):
        classifier.train()
        total_loss = 0.0
        total_examples = 0
        correct = 0
        for x_batch, y_batch in _make_loader(train_x, train_y, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            logits = classifier(x_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size_actual = int(y_batch.shape[0])
            total_loss += float(loss.item()) * batch_size_actual
            total_examples += batch_size_actual
            correct += int((logits.argmax(dim=-1) == y_batch).sum().item())

        logs.append(
            {
                "epoch": float(epoch + 1),
                "cross_entropy": total_loss / max(total_examples, 1),
                "accuracy": correct / max(total_examples, 1),
            }
        )

    pretrain_metrics = {
        "train": _classification_head_metrics(classifier, train_x, train_y, batch_size=batch_size),
        "test": _classification_head_metrics(classifier, test_x, test_y, batch_size=batch_size),
    }
    return classifier.feature_extractor, logs, pretrain_metrics


def benchmark_pretrained_cnn_dirichlet_gp(
    dataset_name: str,
    seeds: list[int] | tuple[int, ...],
    data_path: str | Path | None = None,
    test_size: float = 0.2,
    pretrain_epochs: int = 100,
    pretrain_lr: float = 1e-3,
    pretrain_weight_decay: float = 0.0,
    pretrain_batch_size: int = 128,
    gp_iters: int = 50,
    gp_lr: float = 0.1,
    embedding_dim: int = 10,
    alpha_epsilon: float = 0.01,
    likelihood_samples: int = 256,
    freeze_feature_extractor: bool = False,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.double,
    preprocess: bool = True,
) -> dict[str, Any]:
    dataset_name = dataset_name.lower()
    x_np, y_np = _load_image_dataset_xy(dataset_name, data_path=data_path)
    device = torch.device(device)

    per_seed_results: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for seed in seeds:
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

        if preprocess and dataset_name != "mnist":
            train_x_flat, test_x_flat = _normalize_with_train_stats(train_x_flat, test_x_flat)

        num_classes = int(torch.unique(train_y).numel())
        seed_runtimes: dict[str, float] = {}

        t0 = time.perf_counter()
        pretrained_feature_extractor, pretrain_logs, pretrain_metrics = _pretrain_feature_extractor(
            train_x=train_x_flat,
            train_y=train_y,
            test_x=test_x_flat,
            test_y=test_y,
            image_shape=image_shape,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            n_epochs=pretrain_epochs,
            lr=pretrain_lr,
            weight_decay=pretrain_weight_decay,
            batch_size=pretrain_batch_size,
        )
        seed_runtimes["pretrained_cnn_pretrain"] = time.perf_counter() - t0

        likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
            targets=train_y,
            alpha_epsilon=alpha_epsilon,
            learn_additional_noise=True,
            dtype=dtype,
        ).to(device=device, dtype=dtype)
        model = _DirichletDKLModel(
            train_x=train_x_flat,
            train_y=likelihood.transformed_targets,
            likelihood=likelihood,
            image_shape=image_shape,
            embedding_dim=embedding_dim,
        ).to(device=device, dtype=dtype)
        model.feature_extractor = pretrained_feature_extractor

        if freeze_feature_extractor:
            for parameter in model.feature_extractor.parameters():
                parameter.requires_grad_(False)
            parameter_groups = [
                {"params": model.covar_module.parameters()},
                {"params": model.mean_module.parameters()},
                {"params": model.likelihood.parameters()},
            ]
        else:
            parameter_groups = [
                {"params": model.feature_extractor.parameters()},
                {"params": model.covar_module.parameters()},
                {"params": model.mean_module.parameters()},
                {"params": model.likelihood.parameters()},
            ]

        t0 = time.perf_counter()
        gp_logs = _train_dirichlet_gp(
            model=model,
            likelihood=likelihood,
            train_x=train_x_flat,
            n_iter=gp_iters,
            lr=gp_lr,
            parameter_groups=parameter_groups,
        )
        seed_runtimes["pretrained_cnn_dirichlet_gp_train"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        gp_train_metrics = _classification_metrics(
            model, train_x_flat, train_y, prob_samples=likelihood_samples
        )
        gp_test_metrics = _classification_metrics(
            model, test_x_flat, test_y, prob_samples=likelihood_samples
        )
        seed_runtimes["pretrained_cnn_dirichlet_gp_eval"] = time.perf_counter() - t0
        seed_runtimes["total"] = float(sum(seed_runtimes.values()))

        seed_result = {
            "dataset": dataset_name,
            "seed": int(seed),
            "methods": ["pretrained_cnn_dirichlet_gp"],
            "num_classes": num_classes,
            "runtimes_s": seed_runtimes,
            "metrics": {
                "pretrained_classifier": pretrain_metrics,
                "pretrained_cnn_dirichlet_gp": {
                    "train": gp_train_metrics,
                    "test": gp_test_metrics,
                },
            },
            "logs": {
                "pretrained_classifier": pretrain_logs,
                "pretrained_cnn_dirichlet_gp": gp_logs,
            },
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

        runtime_rows.append({"dataset": dataset_name, "seed": int(seed), **seed_runtimes})

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
        description=(
            "Pre-train the paper CNN feature extractor with cross-entropy and fit a "
            "Dirichlet GP classifier on top of the learned embeddings."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(SUPPORTED_IMAGE_DATASETS),
        choices=SUPPORTED_IMAGE_DATASETS,
        help="Image datasets to benchmark.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds to evaluate.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--pretrain-epochs", type=int, default=75)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-weight-decay", type=float, default=0.0)
    parser.add_argument("--pretrain-batch-size", type=int, default=128)
    parser.add_argument("--gp-iters", type=int, default=50)
    parser.add_argument("--gp-lr", type=float, default=0.1)
    parser.add_argument("--embedding-dim", type=int, default=10)
    parser.add_argument("--alpha-epsilon", type=float, default=0.01)
    parser.add_argument("--likelihood-samples", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--freeze-feature-extractor",
        action="store_true",
        help="Freeze the pretrained CNN during Dirichlet GP fitting.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_outputs/pretrained_cnn_dirichlet_gp_benchmarks",
        help="Directory where metrics/runtime CSVs and JSON summaries are written.",
    )
    parser.add_argument("--coil20-path", default=None)
    parser.add_argument("--mnist-path", default=None)
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
        result = benchmark_pretrained_cnn_dirichlet_gp(
            dataset_name=dataset_name,
            seeds=args.seeds,
            data_path=dataset_paths[dataset_name],
            test_size=args.test_size,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_lr=args.pretrain_lr,
            pretrain_weight_decay=args.pretrain_weight_decay,
            pretrain_batch_size=args.pretrain_batch_size,
            gp_iters=args.gp_iters,
            gp_lr=args.gp_lr,
            embedding_dim=args.embedding_dim,
            alpha_epsilon=args.alpha_epsilon,
            likelihood_samples=args.likelihood_samples,
            freeze_feature_extractor=args.freeze_feature_extractor,
            device=args.device,
        )

        metrics_path = output_dir / f"{dataset_name}_metrics.csv"
        summary_path = output_dir / f"{dataset_name}_metrics_summary.csv"
        runtime_path = output_dir / f"{dataset_name}_runtimes.csv"
        full_json_path = output_dir / f"{dataset_name}_full_results.json"

        result["metrics_df"].to_csv(metrics_path, index=False)
        result["metrics_summary_df"].to_csv(summary_path, index=False)
        result["runtime_df"].to_csv(runtime_path, index=False)
        with full_json_path.open("w") as fp:
            json.dump(result["per_seed_results"], fp, indent=2)

        print(f"[{dataset_name}] Method: pretrained_cnn_dirichlet_gp")
        print(f"[{dataset_name}] Wrote: {metrics_path}")
        print(f"[{dataset_name}] Wrote: {summary_path}")
        print(f"[{dataset_name}] Wrote: {runtime_path}")
        print(f"[{dataset_name}] Wrote: {full_json_path}")
        print(f"\n[{dataset_name}] Metrics preview:")
        print(result["metrics_df"].to_string(index=False))


if __name__ == "__main__":
    main()
