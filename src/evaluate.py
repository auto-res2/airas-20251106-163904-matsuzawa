import json
import os
import sys
from pathlib import Path
from typing import List, Dict

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix
from scipy.stats import binomtest

# -----------------------------------------------------------------------------
#  Helper utilities
# -----------------------------------------------------------------------------

def save_json(obj: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        # Convert to native Python types to handle WandB objects
        json.dump(json.loads(json.dumps(obj, default=str)), f, indent=2)


def plot_learning_curve(df: pd.DataFrame, metric: str, run_id: str, out_dir: Path, ylim: tuple = None) -> Path:
    plt.figure(figsize=(7, 4))
    sns.lineplot(x=df.index, y=df[metric], linewidth=2)
    plt.title(f"{run_id}: {metric}", fontsize=12)
    plt.xlabel("Step", fontsize=11)
    plt.ylabel(metric, fontsize=11)
    plt.grid(True, alpha=0.3)

    # Apply consistent Y-axis limits if provided
    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    fname = out_dir / f"{run_id}_{metric}_learning_curve.pdf"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    return fname


def plot_confusion_matrix(cm: List[List[int]], classes: List[str], run_id: str, out_dir: Path) -> Path:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 11})
    plt.ylabel("True label", fontsize=11)
    plt.xlabel("Predicted label", fontsize=11)
    # Shorten run_id for title if too long
    short_id = run_id if len(run_id) <= 40 else run_id[:37] + "..."
    plt.title(f"{short_id}\nConfusion Matrix", fontsize=10, pad=10)
    plt.tight_layout()
    fname = out_dir / f"{run_id}_confusion_matrix.pdf"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
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
    ax = sns.barplot(data=df, x="run_id", y=metric)
    plt.title(f"Comparison of {metric}")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")

    # Set appropriate Y-axis range to highlight differences
    values = df[metric].values
    val_min, val_max = values.min(), values.max()
    val_range = val_max - val_min

    # For metrics with small ranges (like accuracy), zoom in
    # Only start from 0 if the range is large relative to the values
    if val_range < 0.1 * val_max and val_range > 0:
        # Zoom in to show differences, add 10% padding
        padding = val_range * 0.1
        ax.set_ylim(val_min - padding, val_max + padding)
        # Adjust text position for zoomed view
        text_offset = val_range * 0.02
    else:
        # For large ranges (like runtime), start from 0
        text_offset = val_max * 0.02

    for i, v in enumerate(values):
        plt.text(i, v + text_offset, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fname = out_dir / f"comparison_{metric}_bar_chart.pdf"
    plt.savefig(fname, dpi=300)
    plt.close()
    return fname

# -----------------------------------------------------------------------------
#  Statistical significance test (McNemar's via binom) ------------------------
# -----------------------------------------------------------------------------

def convert_to_list(obj):
    """Convert WandB objects or other iterables to a regular Python list"""
    if isinstance(obj, list):
        return obj

    # Handle WandB SummarySubDict by accessing internal _dict
    if hasattr(obj, '_dict'):
        try:
            keys = sorted([int(k) for k in obj._dict.keys()])
            return [obj._dict[str(k)] for k in keys]
        except (ValueError, TypeError, AttributeError, KeyError):
            return list(obj._dict.values())

    # Check for dict-like interface using callable check
    if callable(getattr(type(obj), 'keys', None)):
        # If it's a dict-like with numeric keys, return values in order
        try:
            keys = sorted([int(k) for k in obj.keys()])
            return [obj[str(k)] for k in keys]
        except (ValueError, TypeError, AttributeError, KeyError):
            try:
                return list(obj.values())
            except (AttributeError, KeyError):
                pass
    # Try to convert as iterable
    try:
        return list(obj)
    except (TypeError, KeyError):
        return [obj]

def mcnemar_pvalue(labels: List[int], preds_a: List[int], preds_b: List[int]) -> float:
    # Ensure all inputs are lists
    labels = convert_to_list(labels)
    preds_a = convert_to_list(preds_a)
    preds_b = convert_to_list(preds_b)

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
    result = binomtest(b10, n=n, p=0.5, alternative="two-sided")
    return result.pvalue

# -----------------------------------------------------------------------------
#  Main evaluation script
# -----------------------------------------------------------------------------

def parse_key_value_args():
    """Parse key=value arguments from command line"""
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key] = value
    return args

def main():
    # Parse key=value arguments
    args = parse_key_value_args()

    results_dir = Path(args['results_dir'])

    # Handle run_ids which is a JSON string
    run_ids: List[str] = json.loads(args['run_ids'])

    # Obtain global WandB config (entity/project)
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    wandb_cfg = OmegaConf.load(cfg_path)
    entity, project = wandb_cfg.wandb.entity, wandb_cfg.wandb.project

    api = wandb.Api()

    run_summaries: Dict[str, Dict] = {}
    generated_files: List[Path] = []

    # First pass: collect all data and compute consistent Y-axis ranges
    run_data = {}
    metric_ranges = {}  # metric -> (min, max)

    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        history = run.history()
        summary = dict(run.summary)
        config = dict(run.config)

        run_data[rid] = {
            "history": history,
            "summary": summary,
            "config": config
        }

        # Track min/max for each metric across all runs
        for metric in ["train_loss", "eval_acc", "glue_accuracy"]:
            if metric in history.columns:
                metric_min = history[metric].min()
                metric_max = history[metric].max()
                if metric not in metric_ranges:
                    metric_ranges[metric] = (metric_min, metric_max)
                else:
                    current_min, current_max = metric_ranges[metric]
                    metric_ranges[metric] = (min(current_min, metric_min), max(current_max, metric_max))

    # Add padding to ranges for better visualization
    for metric in metric_ranges:
        min_val, max_val = metric_ranges[metric]
        range_val = max_val - min_val
        padding = range_val * 0.05  # 5% padding
        metric_ranges[metric] = (min_val - padding, max_val + padding)

    # Second pass: generate plots with consistent scales
    for rid in run_ids:
        history = run_data[rid]["history"]
        summary = run_data[rid]["summary"]
        config = run_data[rid]["config"]

        run_dir = results_dir / rid
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save summary and config
        metrics_path = run_dir / "metrics.json"
        save_json({"summary": summary, "config": config}, metrics_path)
        generated_files.append(metrics_path)

        # Learning curves for common metrics with consistent Y-axis
        for metric in [m for m in ["train_loss", "eval_acc", "glue_accuracy"] if m in history.columns]:
            ylim = metric_ranges.get(metric)
            fname = plot_learning_curve(history, metric, rid, run_dir, ylim=ylim)
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
            summary_a = dict(api.run(f"{entity}/{project}/{rid_a}").summary)
            preds_a = summary_a.get("best_eval_preds")
            labels_a = summary_a.get("best_eval_labels")
            if preds_a is None or labels_a is None:
                continue  # skip if predictions absent
            if base_labels is None:
                base_labels = labels_a
            for rid_b in run_ids[i + 1:]:
                summary_b = dict(api.run(f"{entity}/{project}/{rid_b}").summary)
                preds_b = summary_b.get("best_eval_preds")
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
