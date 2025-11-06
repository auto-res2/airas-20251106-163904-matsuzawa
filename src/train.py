import os
import time
import json
import copy
from collections import defaultdict
from typing import Dict, Any, Tuple, List

import hydra
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, DataCollatorWithPadding,
                          get_scheduler)
import evaluate  # modern replacement for deprecated load_metric
import wandb
import optuna
from sklearn.metrics import confusion_matrix

from src.preprocess import build_datasets
from src.model import build_lora_model, group_parameters


# -----------------------------------------------------------------------------
#  Utility helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Fisher helpers --------------------------------------------------------------

def accumulate_fisher(fisher_dict: Dict[str, torch.Tensor], model: nn.Module, ema_alpha: float):
    """Accumulate EMA of squared gradients (Fisher diagonal proxy)."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ("lora_A" in name) or ("lora_B" in name):
                continue  # skip LoRA params
            if param.grad is None:
                continue
            g2 = param.grad.detach() ** 2
            g2_sum = g2.mean()  # mean over all elements for stability
            if name not in fisher_dict:
                fisher_dict[name] = g2_sum.clone()
            else:
                fisher_dict[name].mul_(ema_alpha).add_(g2_sum, alpha=1 - ema_alpha)


def normalise_fisher(fisher_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """L1-normalise Fisher scores so they sum to 1."""
    total = sum(v.item() for v in fisher_dict.values()) + 1e-12
    return {k: (v.item() / total) for k, v in fisher_dict.items()}


def fisher_weighted_reg(model: nn.Module, fisher_scores: Dict[str, float]) -> torch.Tensor:
    """Compute Fisher-weighted L2 regularisation for all LoRA matrices."""
    reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if ("lora_A" in name) or ("lora_B" in name):
            # Map LoRA param name back to its base matrix name (approximate)
            base_name = name.split(".lora_")[0] + ".weight"
            importance = fisher_scores.get(base_name, 0.0)
            reg_loss = reg_loss + (1.0 - importance) * param.pow(2).sum()
    return reg_loss

# -----------------------------------------------------------------------------
#  Single training routine (optionally after Optuna optimisation)
# -----------------------------------------------------------------------------

def train_single_run(cfg) -> Dict[str, Any]:
    """Performs one full training run according to cfg and returns summary metrics."""
    set_seed(int(cfg.training.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Data ----------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_id, cache_dir=".cache/")
    train_ds, eval_ds, num_labels = build_datasets(cfg, tokenizer)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collator)

    # ---------------- Model & Optimiser --------------------------------------
    model = build_lora_model(cfg, num_labels)
    model.to(device)

    optim_groups = group_parameters(model, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    optimizer = torch.optim.AdamW(optim_groups)

    num_update_steps_per_epoch = max(1, len(train_loader) // cfg.training.gradient_accumulation_steps)
    max_steps = cfg.training.epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(cfg.training.warmup_ratio * max_steps),
        num_training_steps=max_steps,
    )

    metric_acc = evaluate.load("accuracy")
    metric_mcc = evaluate.load("matthews_correlation")

    # Fisher containers
    fisher_raw: Dict[str, torch.Tensor] = {}
    fisher_scores: Dict[str, float] = {}
    fw_enabled = (cfg.method == "proposed_fw_lora")

    scaler = torch.cuda.amp.GradScaler(enabled=str(cfg.training.mixed_precision).lower() in {"fp16", "bf16"})

    global_step = 0
    best_eval_acc = -1.0
    best_preds: List[int] = []
    best_labels: List[int] = []

    start_time = time.time()

    for epoch in range(cfg.training.epochs):
        # ---------------- Training loop --------------------------------------
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(**batch)
                loss = nn.functional.cross_entropy(outputs.logits, labels)
                if fw_enabled and global_step >= cfg.fw_lora.warmup_steps:
                    reg = fisher_weighted_reg(model, fisher_scores)
                    loss = loss + cfg.fw_lora.lambda_fw * reg

            scaler.scale(loss / cfg.training.gradient_accumulation_steps).backward()

            # Fisher accumulation (only during warm-up)
            if fw_enabled and global_step < cfg.fw_lora.warmup_steps:
                accumulate_fisher(fisher_raw, model, cfg.fw_lora.fisher_alpha)

            if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()

            preds = outputs.logits.argmax(dim=-1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item() * labels.size(0)
            global_step += 1

            # Trial-mode speed-up: only two batches
            if cfg.mode == "trial" and step >= 1:
                break

        # Freeze Fisher once warm-up finished
        if fw_enabled and (not fisher_scores) and global_step >= cfg.fw_lora.warmup_steps:
            fisher_scores = normalise_fisher(fisher_raw)
            fisher_raw.clear()

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)

        # ---------------- Evaluation ----------------------------------------
        model.eval()
        eval_loss_total, eval_correct, eval_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for step, batch in enumerate(eval_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    outputs = model(**batch)
                    loss = nn.functional.cross_entropy(outputs.logits, labels)
                eval_loss_total += loss.item() * labels.size(0)
                preds = outputs.logits.argmax(dim=-1)
                eval_correct += (preds == labels).sum().item()
                eval_total += labels.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                if cfg.mode == "trial" and step >= 1:
                    break

        eval_loss = eval_loss_total / max(1, eval_total)
        eval_acc = eval_correct / max(1, eval_total)
        metric_acc.add_batch(predictions=all_preds, references=all_labels)
        metric_mcc.add_batch(predictions=all_preds, references=all_labels)
        glue_accuracy = metric_acc.compute()["accuracy"]
        glue_mcc = metric_mcc.compute()["matthews_correlation"]

        # ---------------- WandB Logging -------------------------------------
        if cfg.wandb.mode != "disabled":
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "glue_accuracy": glue_accuracy,
                "glue_mcc": glue_mcc,
                "lr": lr_scheduler.get_last_lr()[0],
            }, step=global_step)

        # Track best
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_preds = all_preds.copy()
            best_labels = all_labels.copy()

    runtime = time.time() - start_time

    summary = {
        "best_eval_acc": best_eval_acc,
        "runtime_sec": runtime,
        "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    if cfg.wandb.mode != "disabled":
        # Confusion matrix of best epoch
        cm = confusion_matrix(best_labels, best_preds).tolist()
        wandb.summary["confusion_matrix"] = cm
        wandb.summary["best_eval_preds"] = best_preds  # enable external reconstruction
        wandb.summary["best_eval_labels"] = best_labels
        for k, v in summary.items():
            wandb.summary[k] = v
    return summary

# -----------------------------------------------------------------------------
#  Optuna objective ------------------------------------------------------------
# -----------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    # Sample hyper-parameters as described in cfg.optuna.search_space
    for hp, hp_conf in cfg.optuna.search_space.items():
        if hp_conf.type == "loguniform":
            sampled = trial.suggest_float(hp, hp_conf.low, hp_conf.high, log=True)
        elif hp_conf.type == "uniform":
            sampled = trial.suggest_float(hp, hp_conf.low, hp_conf.high)
        elif hp_conf.type == "int":
            sampled = trial.suggest_int(hp, hp_conf.low, hp_conf.high, step=hp_conf.step)
        elif hp_conf.type == "categorical":
            sampled = trial.suggest_categorical(hp, hp_conf.choices)
        else:
            raise ValueError(f"Unknown Optuna hp type: {hp_conf.type}")
        # Assign sampled value
        if hp in cfg.training:
            cfg.training[hp] = sampled
        elif hp in cfg.fw_lora:
            cfg.fw_lora[hp] = sampled
        elif hp == "lora_rank":
            cfg.model.lora.rank = int(sampled)
        else:
            cfg[hp] = sampled

    # Fast trial run: 1 epoch, wandb disabled
    cfg.training.epochs = 1
    cfg.mode = "trial"
    cfg.wandb.mode = "disabled"

    metrics = train_single_run(cfg)
    return metrics["best_eval_acc"]

# -----------------------------------------------------------------------------
#  Hydra entry-point -----------------------------------------------------------
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):  # noqa: C901  complexity fine for research script
    """Main launcher for a single training job. Supports trial/full + Optuna."""
    orig_cwd = hydra.utils.get_original_cwd()
    run_cfg_path = os.path.join(orig_cwd, "config", "runs", f"{cfg.run}.yaml")
    if not os.path.exists(run_cfg_path):
        raise FileNotFoundError(f"Run config file not found: {run_cfg_path}")

    run_cfg = OmegaConf.load(run_cfg_path)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, run_cfg)

    # Mode specific overrides -------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    # ---------------- WandB initialisation -----------------------------------
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.get("run_id", cfg.run),  # fall-back to short run
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )
        print(f"WandB URL: {wandb.run.get_url()}")

    # ---------------- Optuna hyper-parameter search --------------------------
    if int(cfg.optuna.n_trials) > 0:
        study = optuna.create_study(direction=cfg.optuna.direction)
        study.optimize(lambda t: optuna_objective(t, cfg), n_trials=int(cfg.optuna.n_trials))
        best_params = study.best_params
        print("Optuna best parameters:", best_params)
        # Merge best params back into cfg
        for k, v in best_params.items():
            if k in cfg.training:
                cfg.training[k] = v
            elif k in cfg.fw_lora:
                cfg.fw_lora[k] = v
            elif k == "lora_rank":
                cfg.model.lora.rank = int(v)
            else:
                cfg[k] = v

    # ---------------- Final training run -------------------------------------
    summary_metrics = train_single_run(cfg)
    print(json.dumps(summary_metrics, indent=2))

    if cfg.wandb.mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()
