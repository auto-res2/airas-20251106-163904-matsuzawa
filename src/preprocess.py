from typing import Tuple, List

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase


SENTENCE_KEYS = {
    # dataset subset : feature key
    "sst2": "sentence",
    "cola": "sentence",
}


def build_datasets(cfg, tokenizer: PreTrainedTokenizerBase) -> Tuple[Dataset, Dataset, int]:
    """Load and tokenise datasets as specified in cfg.

    Returns
    -------
    train_ds, eval_ds, num_labels
    """
    name: str = cfg.dataset.name
    subset: str = cfg.dataset.get("subset")
    cache_dir = ".cache/"

    raw_dataset = load_dataset(name, subset, cache_dir=cache_dir)

    train_split = cfg.dataset.split
    eval_split = cfg.dataset.eval_split

    train_ds = raw_dataset[train_split]
    eval_ds = raw_dataset[eval_split]

    text_key = SENTENCE_KEYS.get(subset, "sentence")

    def tokenize_fn(examples):
        texts: List[str] = examples[text_key]
        tokenised = tokenizer(
            texts,
            max_length=int(cfg.dataset.max_length),
            padding=cfg.dataset.padding,
            truncation=cfg.dataset.truncation,
        )
        return tokenised

    # Preserve the original label column but remove others we don't need
    remove_cols_train = [c for c in train_ds.column_names if c not in {text_key, "label"}]
    remove_cols_eval = [c for c in eval_ds.column_names if c not in {text_key, "label"}]

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=remove_cols_train)
    eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=remove_cols_eval)

    # Rename label -> labels for transformers' default Trainer compatibility
    train_ds = train_ds.rename_column("label", "labels")
    eval_ds = eval_ds.rename_column("label", "labels")

    num_labels = raw_dataset[train_split].features["label"].num_classes

    # Ensure PyTorch tensors
    train_ds.set_format(type="torch")
    eval_ds.set_format(type="torch")

    return train_ds, eval_ds, num_labels
