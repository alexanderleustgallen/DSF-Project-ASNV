"""Load the final dataset and the fixed train/test split.

Every model notebook uses `get_split()`, so all models train and test on exactly the
same rows. The split itself is created once in notebook 02 with `make_split()`.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src import paths

TARGET = "tm"
SEED = 42


def load_dataset() -> pd.DataFrame:
    """Return data/processed/dataset.parquet: id, organism, ogt, emb_0..emb_479, tm."""
    return pd.read_parquet(paths.DATASET)


def embedding_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("emb_")]


def feature_columns(df: pd.DataFrame, use_ogt: bool = True, use_embeddings: bool = True) -> list[str]:
    """Default: ogt + all embedding columns (481 features)."""
    columns = ["ogt"] if use_ogt else []
    return columns + (embedding_columns(df) if use_embeddings else [])


def make_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED) -> pd.DataFrame:
    """Random split, stratified by organism. Returns id -> 'train' / 'test'.

    Called once in notebook 02; the result is saved to data/processed/split.csv.
    """
    train_ids, test_ids = train_test_split(
        df["id"], test_size=test_size, random_state=seed, stratify=df["organism"]
    )
    split = pd.concat(
        [pd.DataFrame({"id": train_ids, "set": "train"}), pd.DataFrame({"id": test_ids, "set": "test"})]
    )
    return split.sort_values("id", ignore_index=True)


def load_split() -> pd.DataFrame:
    return pd.read_csv(paths.SPLIT)


def get_split(use_ogt: bool = True, use_embeddings: bool = True):
    """Return X_train, X_test, y_train, y_test from the fixed split.

    X keeps the dataset's `id` as index, so predictions can be matched back to proteins.
    """
    df = load_dataset().merge(load_split(), on="id", validate="one_to_one").set_index("id")
    features = feature_columns(df, use_ogt, use_embeddings)
    train, test = df[df["set"] == "train"], df[df["set"] == "test"]
    return train[features], test[features], train[TARGET], test[TARGET]
