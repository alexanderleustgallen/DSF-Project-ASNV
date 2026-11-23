# Instructions for AI agents

Read this before changing anything in this repository. [README.md](README.md) explains the project; this file explains how to work in it.

## Project in one paragraph

A student team predicts protein melting temperature (Tm) and compares several models on **one shared dataset**: one row per protein with `id`, `organism`, `ogt` (optimal growth temperature of the organism), `emb_0` … `emb_479` (ESM-2 35M embedding) and the target `tm`. That gives 481 features. Every model uses the same fixed train/test split.

## Where things live

| Path | Content | Who changes it |
| --- | --- | --- |
| `notebooks/0X_*.ipynb` | The work: code and explanations, one notebook per task | The notebook's owner |
| `src/paths.py` | All file paths | Team agreement only |
| `src/data.py` | `load_dataset()`, `make_split()`, `get_split()` | Team agreement only |
| `src/evaluate.py` | `compute_metrics()`, `save_results()`, `load_all_results()` | Team agreement only |
| `scripts/` | Data pipeline (Meltome download, ESM-2 embeddings), called from notebooks 01 and 02 | Team agreement only |
| `data/raw/ogt.csv` | OGT per organism, with sources | Notebook 01 |
| `data/processed/split.csv` | The fixed train/test split | Nobody, once created |
| `results/` | Metrics, predictions, figures written by the notebooks | Notebooks |

## How to work on a notebook

1. **Only edit the notebook you were asked to work on.** Two people editing the same notebook cause merge conflicts that git cannot resolve.
2. **Follow the notebook's structure.** Each notebook already has headings, instructions and empty code cells. Fill in the code cell below each instruction; add cells where needed, but keep the headings.
3. **Explain before you code.** Every code cell gets a short markdown cell above it saying what it does and why. Teammates read the notebooks to learn.
4. **Keep it simple.** Straightforward pandas and scikit-learn code that a student can follow, no clever abstractions. Put shared logic in `src/` only after the team agrees.
5. **Write decisions down** in the "Notes and decisions" section at the end of the notebook.
6. **Tick the "Done when" checklist** at the top once each item is true.

## Rules that keep results comparable

- **Paths:** use `src.paths`, never hard-coded or absolute paths.
- **Split:** get data with `src.data.get_split()`. Never create another split, and never overwrite `data/processed/split.csv`.
- **Test set:** use it once per model, for the final prediction. Choose hyperparameters by cross-validation or a validation part of the training set.
- **Results:** save every model with `src.evaluate.save_results(model_name, y_test, y_pred, params)`. Use the model names given in the notebooks (`baseline_mean`, `baseline_ogt`, `ridge`, `ridge_no_ogt`, `random_forest`, `xgboost`, `mlp`); new models get a short snake_case name.
- **Seeds:** use 42 everywhere and record parameters in `params`.
- **Data:** never invent data. OGT values need a real source (URL or DOI) in `data/raw/ogt.csv`.
- **Embeddings:** do not recompute them if `data/interim/embeddings.npy` exists (about 3.6 hours on CPU). No fine-tuning of ESM-2.

## Git

- Never commit to `main` directly: work on a branch and open a pull request.
- Do not commit large data files (`.parquet`, `.npy`); `.gitignore` already excludes them. They are shared via the team drive.
- Clear notebook outputs before committing (`nbstripout --install` does this automatically).
- Commit messages: short, imperative, with a type prefix, e.g. `feat(04): add ridge model`.
