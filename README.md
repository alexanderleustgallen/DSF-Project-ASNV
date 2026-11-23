# Protein Tm Prediction: One Dataset, Many Models

Predict a protein's melting temperature (Tm) from its ESM-2 embedding and the optimal growth temperature (OGT) of its organism, then compare several models on the same dataset.

**Timeline:** 2026-11-09 to 2026-11-27, team of four.

> **Changed on 2026-11-23.** The two-stage predictor with an ensemble is dropped, and with it the old briefing (`PROJEKTBEESCHREIBUNG.md`, still in the git history). The work now happens in Jupyter notebooks: one shared dataset is built once, and every model is trained and tested on it with the same split. The Meltome dataset and the ESM-2 embeddings built so far stay valid.

## Pipeline

```
Meltome Atlas ──┐
UniProt ────────┼──► 01 data collection ──► 02 build dataset ──► dataset.parquet + split.csv
OGT sources ────┘                         (ESM-2 embeddings)             │
                                                                         ├──► 03 EDA
                                                                         ├──► 04 linear models  ──┐
                                                                         ├──► 05 tree models    ──┼──► 07 comparison
                                                                         └──► 06 neural network ──┘
```

## Dataset

`data/processed/dataset.parquet`, one row per protein:

| Column | Type | Description |
| --- | --- | --- |
| `id` | str | UniProt accession, unique |
| `organism` | str | Source organism. Not a feature: used for analysis only |
| `ogt` | float | Optimal growth temperature of the organism in °C. Feature |
| `emb_0` … `emb_479` | float | Mean-pooled ESM-2 35M embedding of the sequence. Features |
| `tm` | float | Melting temperature in °C. Target |

That makes **481 features** (`ogt` + 480 embedding dimensions). The raw sequence is not part of the dataset; it stays in `data/interim/meltome.parquet`.

`data/processed/split.csv` assigns every `id` to `train` or `test`. It is created once in notebook 02 (random 80/20, stratified by organism, seed 42) and used by every model.

The source data has 33,904 proteins from 13 organisms. Median Tm per organism ranges from 41 °C (*C. elegans*) to 85 °C (*T. thermophilus*).

## Notebooks

| Notebook | Content | Writes | Owner |
| --- | --- | --- | --- |
| [01_data_collection](notebooks/01_data_collection.ipynb) | Meltome + UniProt, OGT per organism | `data/interim/meltome.parquet`, `data/raw/ogt.csv` | TBD |
| [02_build_dataset](notebooks/02_build_dataset.ipynb) | Merge, clean, ESM-2 embeddings, split | `data/processed/dataset.parquet`, `split.csv` | TBD |
| [03_eda](notebooks/03_eda.ipynb) | Distributions, Tm vs. OGT, embedding PCA | `results/figures/` | TBD |
| [04_linear_models](notebooks/04_linear_models.ipynb) | Baselines, Ridge, Lasso | `results/metrics/`, `results/predictions/` | TBD |
| [05_tree_models](notebooks/05_tree_models.ipynb) | Random forest, XGBoost | `results/metrics/`, `results/predictions/` | TBD |
| [06_neural_network](notebooks/06_neural_network.ipynb) | MLP | `results/metrics/`, `results/predictions/` | TBD |
| [07_model_comparison](notebooks/07_model_comparison.ipynb) | All models side by side | `results/figures/` | TBD |

Notebooks 03 to 06 only need `dataset.parquet` and `split.csv`, so they can be worked on in parallel once notebook 02 has run.

## Repository structure

```
├── notebooks/          the seven notebooks above: code and explanations
├── src/                small shared helpers, imported by the notebooks
│   ├── paths.py        all file paths
│   ├── data.py         load_dataset(), make_split(), get_split()
│   └── evaluate.py     compute_metrics(), save_results(), load_all_results()
├── scripts/            data pipeline scripts, called from notebooks 01 and 02
│   ├── build_meltome.py
│   └── embed_esm.py
├── data/
│   ├── raw/            downloads; ogt.csv is checked in
│   ├── interim/        meltome.parquet, embeddings (not in git)
│   └── processed/      dataset.parquet (not in git), split.csv (in git)
├── results/
│   ├── metrics/        one JSON per model (in git)
│   ├── predictions/    one CSV of test predictions per model (not in git)
│   └── figures/
├── presentation/       final slides and report
└── requirements.txt
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install            # strips notebook outputs on commit, avoids merge conflicts
jupyter lab
```

Large files are not in git. Copy them from the team drive into `data/interim/`:

- `meltome.parquet` (or rebuild it in notebook 01, takes a minute)
- `embeddings.npy`, `embeddings_index.parquet`, `embeddings.run.json` (rebuilding takes about 3.6 hours on CPU)

`data/interim/meltome.manifest.json` holds the expected `content_sha256` of `meltome.parquet`; notebook 02 also checks that the embeddings match the sequences.

## Working rules

1. **One owner per notebook.** Git cannot merge two people's edits to the same notebook.
2. **Every model uses `get_split()` and `save_results()`** from `src/`. That keeps the split and the metrics identical, so the results are comparable.
3. **The split is fixed.** `split.csv` is only recreated if the whole team agrees.
4. **No tuning on the test set.** Hyperparameters are chosen by cross-validation on the training set.
5. **Write decisions down** in the "Notes and decisions" section at the end of each notebook.
6. **Reproducible:** fixed seeds, recorded in `params` when calling `save_results()`.

## Metrics

RMSE in °C (primary), MAE, R² and Spearman correlation, all on the same test set. `baseline_mean` (training mean) and `baseline_ogt` (linear model on OGT only) are the reference points every model has to beat.

## Status

| Part | Status |
| --- | --- |
| Meltome + UniProt data | done, `scripts/build_meltome.py` |
| ESM-2 35M embeddings | done, `scripts/embed_esm.py` |
| OGT per organism | open, fill `data/raw/ogt.csv` |
| Dataset and split | open, notebook 02 |
| Models | open, notebooks 04 to 06 |
| Comparison, presentation | open |

## Open points

- **OGT sources** for all 13 organisms, including animals and plants (body or growth temperature).
- **Lysate vs. cell:** all human rows are `cell` measurements, all others `lysate`. Keep both or drop one?
- **OGT is constant per organism.** With only 13 organisms, `ogt` takes 13 values and partly encodes organism identity. A random split lets models use that; a split by organism would test generalization to unseen organisms. Decide which question the comparison answers.
- **Model list:** which variants per notebook, and whether to also run each model without `ogt`.
