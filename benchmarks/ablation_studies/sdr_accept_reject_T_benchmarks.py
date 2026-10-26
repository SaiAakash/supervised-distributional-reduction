"""
Test the accept/reject safeguard on the SDR coupling update (COIL-20, SNAREseq).

The transport-plan subproblem in SDR's BCD loop is solved by (semi-relaxed, optionally
entropic) fused Gromov-Wasserstein. That solver is a descent step only in exact arithmetic,
so `SDR._update_T` now scores every candidate plan against the objective the solver actually
minimizes (`SDR._t_gate_objective`) and keeps it only on a strict decrease. Rejected updates
are counted in `SDR.n_T_rejections` and itemized in `SDR.T_update_history`.

This benchmark sweeps the number of prototypes m = output_sam over three seeds and, for each
configuration, fits SDR under three arms:

  * arm "J"   -- accept_reject_T=True, t_gate="J": gate on the *complete* SDR objective
                 J(T) = F(T) - lambda_hsic * CKA(Z, Y(T))
  * arm "F"   -- accept_reject_T=True, t_gate="F": gate on the solver's own objective F only
  * arm "off" -- accept_reject_T=False: every candidate plan kept (previous behaviour)

The three arms are what make this a test rather than a log dump. "off" establishes how many
candidate updates were non-decreasing in the first place; "F" vs "J" isolates the term the
solver ignores: because CKA depends on T whenever m < n, a step can decrease F while increasing
J, so only the "J" arm makes the T-step monotone in the objective that defines the method. The
per-run counter `F_decreased_but_rejected` counts exactly those CKA-vetoed steps.

Rejections are itemized by reason so that a converged solver is not mistaken for a failing one:

  * "no_decrease"  -- the candidate strictly increased the objective (a genuine uphill step
                      that the safeguard blocked)
  * "no_progress"  -- the solver returned the same plan to numerical precision (it has
                      converged; rejecting it is a no-op)
  * "non_finite"   -- the candidate produced a NaN/inf objective

Every hyperparameter other than `accept_reject_T` and `output_sam` is imported from
method_comparison, so results are directly comparable to the joint/sequential benchmarks
(same affinities, loss, HSIC kernels, spectral T initialisation, lambda_hsic and per-dataset
perplexity/learning rate).

Reported metrics: homogeneity, k-means NMI, silhouette (plus the plan-assignment NMI), each on
the 0-100 scale used throughout `src.scores`.
"""

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from src.affinities import SymmetricEntropicAffinity
from src.sdr import SDR

from benchmarks.benchmarks_clustdr.classification_joint_benchmarks import _preprocess_features
from benchmarks.benchmarks_rebuttal.method_comparison import (
    BASE_AFFINITY_KWARGS,
    DATASET_AFFINITY_KWARGS,
    DATASET_LOADERS,
    FGW_LOSS_KWARGS,
    HSIC_KWARGS,
    _compute_cka_score,
    _compute_joint_metrics,
    _final_loss_terms,
    _set_seed,
)


DEFAULT_DATASETS = ("coil20", "snareseq")
DEFAULT_OUTPUT_SAMS = (20, 60, 100, 140, 180)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_ARMS = ("J", "F", "off")

REJECTION_REASONS = ("no_decrease", "no_progress", "non_finite")

# arm -> (accept_reject_T, t_gate)
ARM_SETTINGS = {
    "J": (True, "J"),
    "F": (True, "F"),
    "off": (False, "J"),  # t_gate is irrelevant when the safeguard is off
}


def _make_sdr(
    dataset_name: str,
    seed: int,
    output_sam: int,
    output_dim: int,
    device: str,
    max_iter: int,
    max_iter_outer: int,
    alpha: float,
    lambda_hsic: float,
    accept_reject_T: bool,
    t_gate: str,
    init_T: str,
) -> SDR:
    affinity_kwargs = DATASET_AFFINITY_KWARGS.get(dataset_name)
    if affinity_kwargs is None:
        raise ValueError(f"Unsupported dataset for affinity configuration: {dataset_name}")

    # BASE_AFFINITY_KWARGS pins init_T="spectral"; override it so the coupling
    # initialisation can be swept.
    base_kwargs = deepcopy(BASE_AFFINITY_KWARGS)
    base_kwargs["init_T"] = init_T

    return SDR(
        **base_kwargs,
        **deepcopy(HSIC_KWARGS),
        **deepcopy(FGW_LOSS_KWARGS),
        affinity_data=SymmetricEntropicAffinity(**affinity_kwargs, verbose=False),
        objective="sdr",
        seed=seed,
        output_sam=output_sam,
        output_dim=output_dim,
        device=device,
        dtype=torch.double,
        max_iter=max_iter,
        max_iter_outer=max_iter_outer,
        alpha=alpha,
        lambda_hsic=lambda_hsic,
        accept_reject_T=accept_reject_T,
        t_gate=t_gate,
    )


def _summarize_T_history(method: SDR) -> dict[str, float | int]:
    """Condense `SDR.T_update_history` into per-run counters."""
    history = getattr(method, "T_update_history", []) or []

    counts = {f"rejected_{reason}": 0 for reason in REJECTION_REASONS}
    for entry in history:
        if not entry["accepted"]:
            key = f"rejected_{entry['reason']}"
            counts[key] = counts.get(key, 0) + 1

    # Steps whose kept plan did NOT decrease the gate objective. With the safeguard on this
    # must be 0 by construction; with it off it counts the non-descent steps that survived.
    violations = sum(
        1
        for entry in history
        if entry["fgw_after"] > entry["fgw_before"] + 1e-9 * abs(entry["fgw_before"])
    )
    # Total uphill movement that survived, in objective units (0.0 when the safeguard holds).
    violation_magnitude = sum(
        max(0.0, entry["fgw_after"] - entry["fgw_before"]) for entry in history
    )

    # Steps the solver improved (F went down) that the gate blocked anyway -- i.e. the CKA
    # supervision term is what vetoed them. Non-zero only under the J-gate.
    cka_vetoed = sum(1 for e in history if e.get("F_decreased_but_rejected"))

    return {
        "n_T_updates": int(getattr(method, "n_T_updates", 0)),
        "n_T_rejections": int(getattr(method, "n_T_rejections", 0)),
        "n_T_accepted": int(sum(1 for e in history if e["accepted"])),
        "cka_vetoed_steps": int(cka_vetoed),
        **counts,
        "fgw_monotonicity_violations": violations,
        "fgw_violation_magnitude": float(violation_magnitude),
        "fgw_first": float(history[0]["fgw_before"]) if history else float("nan"),
        "fgw_last": float(history[-1]["fgw_after"]) if history else float("nan"),
        "mean_delta_T": (
            float(sum(e["delta_T"] for e in history) / len(history)) if history else float("nan")
        ),
    }


def _run_single(
    dataset_name: str,
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
    arm: str,
    init_T: str,
) -> dict[str, float | int | str]:
    accept_reject_T, t_gate = ARM_SETTINGS[arm]

    _set_seed(seed)
    method = _make_sdr(
        dataset_name=dataset_name,
        seed=seed,
        output_sam=output_sam,
        output_dim=output_dim,
        device=device,
        max_iter=max_iter,
        max_iter_outer=max_iter_outer,
        alpha=alpha,
        lambda_hsic=lambda_hsic,
        accept_reject_T=accept_reject_T,
        t_gate=t_gate,
        init_T=init_T,
    )

    start = time.perf_counter()
    z, _ = method.fit_transform(x, y)
    runtime_s = time.perf_counter() - start

    metrics = _compute_joint_metrics(x=x, y=y, t=method.T.detach(), z=z.detach())
    cka_score = _compute_cka_score(method=method, y=y, z=z)
    losses = _final_loss_terms(method)
    t_stats = _summarize_T_history(method)

    return {
        "dataset": dataset_name,
        "method": "sdr",
        "arm": arm,
        "accept_reject_T": accept_reject_T,
        "t_gate": t_gate,
        "init_T": init_T,
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
        "kmeans_nmi": metrics["kmeans"],
        "silhouette": metrics["silhouette"],
        "cka_score": cka_score,
        **t_stats,
        "final_objective": losses["final_objective"],
        "final_structural_loss": losses["final_structural_loss"],
        "final_supervision_loss": losses["final_supervision_loss"],
        "fit_transform_runtime_s": runtime_s,
        "inner_steps": len(getattr(method, "inner_loss_history", []) or []),
        "outer_steps": len(getattr(method, "outer_loss_history", []) or []),
    }, method


def _aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby(["dataset", "arm", "output_sam"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            homogeneity_mean=("homogeneity", "mean"),
            homogeneity_std=("homogeneity", "std"),
            kmeans_nmi_mean=("kmeans_nmi", "mean"),
            kmeans_nmi_std=("kmeans_nmi", "std"),
            silhouette_mean=("silhouette", "mean"),
            silhouette_std=("silhouette", "std"),
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            n_T_updates_mean=("n_T_updates", "mean"),
            n_T_rejections_mean=("n_T_rejections", "mean"),
            n_T_rejections_std=("n_T_rejections", "std"),
            n_T_rejections_total=("n_T_rejections", "sum"),
            rejected_no_decrease_mean=("rejected_no_decrease", "mean"),
            rejected_no_progress_mean=("rejected_no_progress", "mean"),
            rejected_non_finite_mean=("rejected_non_finite", "mean"),
            cka_vetoed_mean=("cka_vetoed_steps", "mean"),
            cka_vetoed_total=("cka_vetoed_steps", "sum"),
            rejected_no_decrease_total=("rejected_no_decrease", "sum"),
            rejected_no_progress_total=("rejected_no_progress", "sum"),
            rejected_non_finite_total=("rejected_non_finite", "sum"),
            fgw_violations_total=("fgw_monotonicity_violations", "sum"),
            cka_score_mean=("cka_score", "mean"),
            runtime_mean_s=("fit_transform_runtime_s", "mean"),
            active_prototypes_mean=("active_prototypes", "mean"),
        )
        .sort_values(["dataset", "arm", "output_sam"])
    )


def _format_markdown_tables(results_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    """Per-run and aggregated tables, mean +/- std over seeds."""
    lines: list[str] = []

    lines.append("# SDR accept/reject T-step benchmark\n")
    lines.append(
        "Metrics on the 0-100 scale. In the aggregated tables all T-step counts are "
        "**averages per run** (a run = one fit at one seed), since the number of outer BCD "
        "iterations differs between runs and raw totals are therefore not comparable across "
        "rows. `up` = rejections that were genuine uphill steps (`no_decrease`), "
        "`conv` = rejections where the solver returned the same plan (`no_progress`).\n"
    )

    for dataset in sorted(results_df["dataset"].unique()):
        lines.append(f"\n## {dataset}\n")
        lines.append("### Per-run (J-gate)\n")
        lines.append(
            "| m | seed | Homogeneity | k-Means NMI | Silhouette | "
            "T-updates | rejected | up | conv | CKA-vetoed | runtime (s) |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        sub = results_df[(results_df["dataset"] == dataset) & (results_df["arm"] == "J")]
        for _, r in sub.sort_values(["output_sam", "seed"]).iterrows():
            lines.append(
                f"| {int(r['output_sam'])} | {int(r['seed'])} | {r['homogeneity']:.2f} | "
                f"{r['kmeans_nmi']:.2f} | {r['silhouette']:.2f} | {int(r['n_T_updates'])} | "
                f"{int(r['n_T_rejections'])} | {int(r['rejected_no_decrease'])} | "
                f"{int(r['rejected_no_progress'])} | {int(r['cka_vetoed_steps'])} | "
                f"{r['fit_transform_runtime_s']:.1f} |"
            )

        lines.append("\n### Aggregated over seeds (mean +/- std; T-step counts per run)\n")
        lines.append(
            "| m | arm | Homogeneity | k-Means NMI | Silhouette | "
            "T-updates/run | rejections/run | up/run | conv/run | CKA-vetoed/run | violations |"
        )
        lines.append("|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        ssub = summary_df[summary_df["dataset"] == dataset]
        for _, r in ssub.sort_values(["output_sam", "arm"]).iterrows():
            def pm(mean_key: str, std_key: str) -> str:
                std = r[std_key]
                std_txt = "0.00" if pd.isna(std) else f"{std:.2f}"
                return f"{r[mean_key]:.2f} +/- {std_txt}"

            lines.append(
                f"| {int(r['output_sam'])} | {r['arm']} | "
                f"{pm('homogeneity_mean', 'homogeneity_std')} | "
                f"{pm('kmeans_nmi_mean', 'kmeans_nmi_std')} | "
                f"{pm('silhouette_mean', 'silhouette_std')} | "
                f"{r['n_T_updates_mean']:.1f} | "
                f"{pm('n_T_rejections_mean', 'n_T_rejections_std')} | "
                f"{r['rejected_no_decrease_mean']:.2f} | "
                f"{r['rejected_no_progress_mean']:.2f} | "
                f"{r['cka_vetoed_mean']:.2f} | "
                f"{int(r['fgw_violations_total'])} |"
            )

    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the SDR accept/reject T-step safeguard on COIL-20 and SNAREseq."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--output-sams", nargs="+", type=int, default=list(DEFAULT_OUTPUT_SAMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(DEFAULT_ARMS),
        choices=list(ARM_SETTINGS),
        help=(
            "'J' = gate on the complete objective J = F - eta*CKA, "
            "'F' = gate on the solver's objective F only, "
            "'off' = no accept/reject."
        ),
    )
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument(
        "--init-T",
        default="spectral",
        help="Coupling initialisation passed to SDR (e.g. spectral, random, unif, kmeans).",
    )
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--lambda-hsic", type=float, default=1000.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--max-iter-outer", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "benchmark_outputs" / "sdr_accept_reject_T"),
    )
    parser.add_argument("--coil20-path")
    parser.add_argument("--snareseq-path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    unknown_datasets = set(args.datasets) - set(DATASET_LOADERS)
    if unknown_datasets:
        raise ValueError(f"Unsupported datasets requested: {sorted(unknown_datasets)}")

    if not 0.0 < args.alpha < 1.0:
        raise ValueError(
            f"alpha must satisfy 0 < alpha < 1 for the FGW/SDR objective; got {args.alpha}."
        )

    dataset_path_map = {
        "coil20": args.coil20_path,
        "snareseq": args.snareseq_path,
    }

    torch.set_default_dtype(torch.double)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cached_datasets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for dataset_name in args.datasets:
        x, y = DATASET_LOADERS[dataset_name](dataset_path_map[dataset_name])
        x = _preprocess_features(x)
        cached_datasets[dataset_name] = (x.to(args.device), y.to(args.device))

    rows: list[dict[str, float | int | str]] = []
    history_path = results_dir / "T_update_history.jsonl"
    with history_path.open("w") as history_file:
        for dataset_name in args.datasets:
            x, y = cached_datasets[dataset_name]
            for output_sam in args.output_sams:
                if output_sam > x.shape[0]:
                    raise ValueError(
                        f"output_sam={output_sam} exceeds sample count {x.shape[0]} "
                        f"for {dataset_name}."
                    )
                for seed in args.seeds:
                    for arm in args.arms:
                        row, method = _run_single(
                            dataset_name=dataset_name,
                            x=x,
                            y=y,
                            seed=seed,
                            output_sam=output_sam,
                            output_dim=args.output_dim,
                            device=args.device,
                            max_iter=args.max_iter,
                            max_iter_outer=args.max_iter_outer,
                            alpha=args.alpha,
                            lambda_hsic=args.lambda_hsic,
                            arm=arm,
                            init_T=args.init_T,
                        )
                        rows.append(row)

                        # Full per-step audit trail, one JSON object per run.
                        history_file.write(
                            json.dumps(
                                {
                                    "dataset": dataset_name,
                                    "arm": arm,
                                    "output_sam": output_sam,
                                    "seed": seed,
                                    "history": method.T_update_history,
                                }
                            )
                            + "\n"
                        )
                        history_file.flush()

                        print(
                            json.dumps(
                                {
                                    "dataset": dataset_name,
                                    "arm": arm,
                                    "m": output_sam,
                                    "seed": seed,
                                    "homogeneity": row["homogeneity"],
                                    "kmeans_nmi": row["kmeans_nmi"],
                                    "silhouette": row["silhouette"],
                                    "n_T_updates": row["n_T_updates"],
                                    "n_T_rejections": row["n_T_rejections"],
                                    "rejected_no_decrease": row["rejected_no_decrease"],
                                    "rejected_no_progress": row["rejected_no_progress"],
                                    "cka_vetoed_steps": row["cka_vetoed_steps"],
                                    "fgw_violations": row["fgw_monotonicity_violations"],
                                    "runtime_s": row["fit_transform_runtime_s"],
                                }
                            ),
                            flush=True,
                        )

    results_df = pd.DataFrame(rows)
    summary_df = _aggregate_results(results_df)

    raw_path = results_dir / "run_metrics.csv"
    summary_path = results_dir / "summary.csv"
    config_path = results_dir / "config.json"
    table_path = results_dir / "results_tables.md"

    results_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    table_path.write_text(_format_markdown_tables(results_df, summary_df))

    config = {
        "datasets": args.datasets,
        "method": "sdr",
        "objective": "sdr",
        "arms": args.arms,
        "output_sams": args.output_sams,
        "seeds": args.seeds,
        "output_dim": args.output_dim,
        "init_T": args.init_T,
        "alpha": args.alpha,
        "lambda_hsic": args.lambda_hsic,
        "max_iter": args.max_iter,
        "max_iter_outer": args.max_iter_outer,
        "device": args.device,
        "hsic_kwargs": HSIC_KWARGS,
        "fgw_loss_kwargs": FGW_LOSS_KWARGS,
        "dataset_affinity_kwargs": DATASET_AFFINITY_KWARGS,
        "base_affinity_kwargs": {k: str(v) for k, v in BASE_AFFINITY_KWARGS.items()},
    }
    config_path.write_text(json.dumps(config, indent=2))

    # Headline check: each gated arm must leave no surviving non-decreasing T-step in
    # *its own* gated objective.
    for arm in args.arms:
        sub = results_df[results_df["arm"] == arm]
        if sub.empty:
            continue
        print(
            f"\nGate check (arm={arm}): "
            f"{int(sub['fgw_monotonicity_violations'].sum())} surviving non-decreasing T-steps "
            f"across {len(sub)} runs "
            f"({int(sub['n_T_rejections'].sum())} rejections / "
            f"{int(sub['n_T_updates'].sum())} updates; "
            f"{int(sub['cka_vetoed_steps'].sum())} CKA-vetoed)."
        )

    print(f"Saved raw run metrics to {raw_path}")
    print(f"Saved summary metrics to {summary_path}")
    print(f"Saved markdown tables to {table_path}")
    print(f"Saved per-step T history to {history_path}")
    print(f"Saved run configuration to {config_path}")


if __name__ == "__main__":
    main()
