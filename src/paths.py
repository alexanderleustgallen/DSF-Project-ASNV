"""Central file paths. Notebooks import these instead of hard-coding paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

# Notebook 01
MELTOME = INTERIM / "meltome.parquet"
OGT = RAW / "ogt.csv"

# Notebook 02 (embeddings are written by scripts/embed_esm.py)
EMBEDDINGS = INTERIM / "embeddings.npy"
EMBEDDINGS_INDEX = INTERIM / "embeddings_index.parquet"
DATASET = PROCESSED / "dataset.parquet"
SPLIT = PROCESSED / "split.csv"

# Notebooks 04-07
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
PREDICTIONS = RESULTS / "predictions"
FIGURES = RESULTS / "figures"
