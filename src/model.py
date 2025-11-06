from typing import List
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
import torch.nn as nn


def build_lora_model(cfg: DictConfig, num_labels: int) -> nn.Module:
    """Construct a (possibly LoRA-augmented) model as specified by cfg."""
    base_cfg = AutoConfig.from_pretrained(cfg.model.pretrained_id, num_labels=num_labels, cache_dir=".cache/")
    base_model = AutoModelForSequenceClassification.from_pretrained(cfg.model.pretrained_id, config=base_cfg, cache_dir=".cache/")

    if cfg.model.lora.enabled:
        lora_cfg = LoraConfig(
            r=int(cfg.model.lora.rank),
            lora_alpha=int(cfg.model.lora.alpha),
            lora_dropout=float(cfg.model.lora.dropout),
            target_modules=list(cfg.model.lora.target_modules),
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(base_model, lora_cfg)
    else:
        model = base_model

    # Require grads for all params so Fisher can accumulate, but only LoRA + classifier will be optimised.
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model


def group_parameters(model: nn.Module, lr: float, weight_decay: float):
    """Return parameter groups for AdamW: LoRA + classifier trainable."""
    no_decay = ["bias", "LayerNorm.weight"]
    trainable_params = [(n, p) for n, p in model.named_parameters() if ("lora_" in n) or ("classifier" in n)]

    params_with_decay = [p for n, p in trainable_params if not any(nd in n for nd in no_decay)]
    params_without_decay = [p for n, p in trainable_params if any(nd in n for nd in no_decay)]

    return [
        {"params": params_with_decay, "lr": lr, "weight_decay": weight_decay},
        {"params": params_without_decay, "lr": lr, "weight_decay": 0.0},
    ]
