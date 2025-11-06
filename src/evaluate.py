import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix
from scipy.stats import binom_test

# -----------------------------------------------------------------------------
#  Helper utilities
# -----------------------------------------------------------------------------

def save_json(obj: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def plot_learning_curve(df: pd.DataFrame, metric: str, run_id: str, out_dir: Path) -> Path:
    plt.figure(figsize=(7, 4))
    sns.lineplot(x=df.index, y=df[metric])
    plt.title(f"{run_id}: {metric}")
    plt.xlabel("Step")
    plt.ylabel(metric)
    plt.tight_layout()
    fname = out_dir / f"{run_id}_{metric}_learning_curve.pdf"
    plt.savefig(fname)
    plt.close()
    return fname


def plot_confusion_matrix(cm: List[List[int]], classes: List[str], run_id: str, out_dir: Path) -> Path:
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(f"{run_id} Confusion Matrix")
    plt.tight_layout()
    fname = out_dir / f"{run_id}_confusion_matrix.pdf"
    plt.savefig(fname)
    plt.close()
    return fname


def aggregate_metrics(run_summaries: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, float]] = {}
    for run_id, summary in run_summaries.items():
        for k, v in summary.items():
            if v is None:
                continue
            agg.setdefault(k, {})[run_id] = v
    return agg


def plot_bar_comparison(metric_table: Dict[str, Dict[str, float]], metric: str, out_dir: Path) -> Path:
    data = {"run_id": list(metric_table[metric].keys()), metric: list(metric_table[metric].values())}
    df = pd.DataFrame(data)
    plt.figure(figsize=(max(6, len(df) * 1.2), 4))
    sns.barplot(data=df, x="run_id", y=metric)
    plt.title(f"Comparison of {metric}")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(df[metric]):
        plt.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    fname = out_dir / f"comparison_{metric}_bar_chart.pdf"
    plt.savefig(fname)
    plt.close()
    return fname

# -----------------------------------------------------------------------------
#  Statistical significance test (McNemar's via binom) ------------------------
# -----------------------------------------------------------------------------

def mcnemar_pvalue(labels: List[int], preds_a: List[int], preds_b: List[int]) -> float:
    assert len(labels) == len(preds_a) == len(preds_b)
    # b00 correct by none; b10 correct by A only; b01 correct by B only; b11 correct by both
    b10 = 0  # A correct, B wrong
    b01 = 0  # B correct, A wrong
    for y, pa, pb in zip(labels, preds_a, preds_b):
        a_correct = (pa == y)
        b_correct = (pb == y)
        if a_correct and (not b_correct):
            b10 += 1
        elif b_correct and (not a_correct):
            b01 += 1
    n = b10 + b01
    if n == 0:
        return 1.0  # identical predictions
    p = binom_test(b10, n=n, p=0.5, alternative="two-sided")
    return p

# -----------------------------------------------------------------------------
#  Main evaluation script
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Independent evaluation of multiple runs from WandB.")
    parser.add_argument("results_dir", type=str, help="Directory where evaluation artefacts will be stored")
    parser.add_argument("run_ids", type=str, help="JSON list of WandB run IDs, e.g. '[\"run-1\", \"run-2\"]'")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_ids: List[str] = json.loads(args.run_ids)

    # Obtain global WandB config (entity/project)
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    entity, project = cfg.wandb.entity, cfg.wandb.project

    api = wandb.Api()

    run_summaries: Dict[str, Dict] = {}
    generated_files: List[Path] = []

    # ---------------- Per-run processing ------------------------------------
    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        history = run.history()  # DataFrame of logged metrics
        summary = dict(run.summary)
        config = dict(run.config)

        run_dir = results_dir / rid
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save summary and config
        metrics_path = run_dir / "metrics.json"
        save_json({"summary": summary, "config": config}, metrics_path)
        generated_files.append(metrics_path)

        # Learning curves for common metrics
        for metric in [m for m in ["train_loss", "eval_acc", "glue_accuracy"] if m in history.columns]:
            fname = plot_learning_curve(history, metric, rid, run_dir)
            generated_files.append(fname)

        # Confusion matrix if available
        if "confusion_matrix" in summary:
            cm = summary["confusion_matrix"]
            fname = plot_confusion_matrix(cm, classes=["neg", "pos"], run_id=rid, out_dir=run_dir)
            generated_files.append(fname)

        run_summaries[rid] = {
            "best_eval_acc": summary.get("best_eval_acc"),
            "runtime_sec": summary.get("runtime_sec"),
        }

    # ---------------- Aggregated analysis -----------------------------------
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    agg_metrics = aggregate_metrics(run_summaries)
    agg_path = comparison_dir / "aggregated_metrics.json"
    save_json(agg_metrics, agg_path)
    generated_files.append(agg_path)

    # Bar plots for each aggregated metric
    for metric in agg_metrics.keys():
        fname = plot_bar_comparison(agg_metrics, metric, comparison_dir)
        generated_files.append(fname)

    # Pair-wise statistical significance (McNemar) if predictions exist
    if len(run_ids) >= 2:
        sig_results: Dict[str, Dict[str, float]] = {}
        # Load labels once (assume ground truth same across runs)
        base_labels = None
        for i, rid_a in enumerate(run_ids):
            preds_a = api.run(f"{entity}/{project}/{rid_a}").summary.get("best_eval_preds")
            labels_a = api.run(f"{entity}/{project}/{rid_a}").summary.get("best_eval_labels")
            if preds_a is None or labels_a is None:
                continue  # skip if predictions absent
            if base_labels is None:
                base_labels = labels_a
            for rid_b in run_ids[i + 1:]:
                preds_b = api.run(f"{entity}/{project}/{rid_b}").summary.get("best_eval_preds")
                if preds_b is None:
                    continue
                p_val = mcnemar_pvalue(base_labels, preds_a, preds_b)
                sig_results.setdefault(rid_a, {})[rid_b] = p_val
        sig_path = comparison_dir / "significance_tests.json"
        save_json(sig_results, sig_path)
        generated_files.append(sig_path)

    # ---------------- Print all generated paths -----------------------------
    for path in generated_files:
        print(str(path))


if __name__ == "__main__":
    main()
