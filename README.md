# Tm Prediction with a Two-Stage Model and Ensemble

Predict a protein's melting temperature (Tm) from its amino acid sequence, together with the organism it most likely comes from.

**Timeline:** 2026-11-09 to 2026-11-27, team of four.

This README is an overview. The full project briefing, in German, is [PROJEKTBEESCHREIBUNG.md](PROJEKTBEESCHREIBUNG.md). It is the single source of truth: read it before taking on a task, and agree on deviations beforehand.

> **Changed on 2026-11-11.** The project no longer has a research question. Hypotheses H1–H4, the four split conditions and the leakage analysis are dropped. The goal is now a working prediction tool. The dataset and ESM-2 embeddings built so far stay valid.

## Goal

A learning project that ends with a usable model:

- **Input:** a protein's amino acid sequence.
- **Output:** predicted Tm in °C with an error estimate, plus the predicted source organism.

```
python predict.py --sequence MKTAYIAKQR...
```

It does not matter whether similar tools exist. The point is to build several models, compare them, and combine them into one tool.

## Status

| Part | Status | File |
| --- | --- | --- |
| Dataset | done | `data/meltome.parquet`, built by `scripts/build_meltome.py` |
| ESM-2 35M embeddings | done | `cache/embeddings.npy`, built by `scripts/embed_esm.py` |
| Split and folds | open | `splits/` |
| Stage 1, Tm models, ensemble | open | |
| `predict.py`, app | open | |

The dataset has 33,904 proteins from 13 organisms in the Meltome Atlas. Median Tm per organism ranges from 41 °C (*C. elegans*) to 85 °C (*Thermus thermophilus*). All `cell` rows are human, and human only appears as `cell`, so measurement type and organism cannot be separated.

## Architecture

```
Sequence ──► Stage 1: organism classifier (ESM ──► 13 probabilities)
   │                          │
   ├──► comp + organism ──► Ridge ────────► Tm₁ ─┐
   ├──► ESM  + organism ──► Ridge / MLP ──► Tm₂ ─┼──► Ensemble ──► Tm ± error
   └──► one-hot ──► CNN (optional) ───────► Tm₃ ─┘
```

### Stage 1: organism classifier

Logistic regression (optionally a small MLP) on the 480-dimensional ESM-2 embedding. It outputs one probability per organism and uses class weights, because class sizes range from 253 to 9,791 proteins.

### Stage 2: Tm models

| Name | Features | Model | Status |
| --- | --- | --- | --- |
| `comp` | 20 amino acid frequencies + length + organism probabilities | Ridge | required |
| `esm` | ESM-2 embedding + organism probabilities | Ridge, optionally MLP | required |
| `cnn` | one-hot sequence only | small 1D CNN from scratch | optional, decided at Gate 1 |

### Ensemble

A Ridge model learns how much each Tm model counts. If the CNN is dropped, the ensemble combines `comp` and `esm`.

### Baselines

| Name | Prediction |
| --- | --- |
| `mean` | Training set mean |
| `organism` | Median Tm of the true source organism, ignoring the sequence |

### Out-of-fold predictions

Stage 2 and the ensemble are trained on predictions from other models. These must be out-of-fold: the prediction for a training row in fold k comes from a model trained without fold k. Otherwise stage 1 is overconfident on its own training data and the later stages rely on it too much.

## Evaluation

- **One fixed test split:** sequences are clustered with MMseqs2 (proposed threshold 30 % identity). 80 % of clusters go to training, 20 % to test. Training has five cluster-based folds.
- **Tm metrics:** RMSE in °C (primary), MAE, Spearman. The error shown by `predict.py` is the final ensemble's test RMSE.
- **Stage 1 metrics:** accuracy, balanced accuracy, confusion matrix.
- **Comparisons:** each model against the ensemble; with and without the organism feature; all models against both baselines; ensemble weights.

## Interfaces

These formats are fixed at Gate 1 and only change afterwards if everyone agrees. Details are in the briefing, section 5.

| File | Content |
| --- | --- |
| `data/meltome.parquet` | `id`, `sequence`, `tm`, `organism`, `condition`; row numbers are the index space for everything else |
| `splits/main.json` | `identity`, `seed`, `train`, `train_fold`, `test` |
| `splits/final.json` | same format, all rows in `train`, empty `test`, for the final model |
| `predictions/{model}.parquet` | `row`, `part` (`oof` or `test`), `tm_pred` |
| `predictions/organism.parquet` | `row`, `part`, one `p_{organism}` column per class |
| `results/results.csv` | one row per Tm model: `model, rmse, mae, spearman, n_train, n_test, seed` |
| `results/classifier.csv` | stage 1 metrics |

## Hard rules

1. **Split and folds belong to role A.** Model code loads `splits/*.json` and never computes splits.
2. **The test set is only used for the final evaluation.** Every hyperparameter search runs over the training folds.
3. **Stage 2 and the ensemble are trained only on out-of-fold predictions.**
4. **No fine-tuning of ESM-2.**
5. **Embeddings are computed once and cached.**
6. **Nothing new after Gate 2 (2026-11-20).** Week 3 is for the app, figures and the report.
7. **Every result is reproducible:** fixed seed, recorded in the result row.

## Roles

| Role | Responsibility | Delivers |
| --- | --- | --- |
| A | Data, `lysate`/`cell` decision, MMseqs2 clustering, split and folds | `data/meltome.parquet`, `splits/*.json` |
| B | Stage 1, CNN | `predictions/organism.parquet`, `predictions/cnn.parquet`, `results/classifier.csv` |
| C | `comp` and `esm` models, ensemble, evaluation, `predict.py` | `predictions/*.parquet`, `results/results.csv`, `predict.py` |
| D | App, figures, report | `app.py`, `figures/`, `paper/` |

D supports A and C in weeks 1 and 2, then works independently from Gate 2 onwards.

## Schedule

| Period | A | B | C | D |
| --- | --- | --- | --- | --- |
| Nov 9–13 | Data (done), split and folds | Stage 1 | `comp`, `esm`, baselines | Supports A and C |
| Nov 16–20 | Final folds, reproduction | CNN | Ensemble, `predict.py` | App skeleton, first figures |
| Nov 23–27 | Verify reproduction | Verify reproduction | Verify reproduction | App, figures, report |

- **Gate 1, Fri Nov 13:** the split exists, and stage 1, `comp`, `esm` and both baselines have test numbers. Decide whether the CNN stays in.
- **Gate 2, Fri Nov 20:** the ensemble and `predict.py` run on the final model. Feature freeze from here on.
- **Submission, Fri Nov 27.**

## Compute

A normal laptop is enough. The ESM-2 35M embeddings took 3.6 hours on CPU and are done. Ridge models, stage 1 and the ensemble take seconds to minutes. A small CNN takes roughly 20 minutes to 1.5 hours per training run on all data. If that is too slow, shrink it, run it overnight, use a free Colab GPU, or drop it.

## Reproducing the data

Requires Python with pandas and pyarrow, plus torch and fair-esm for the embeddings.

```
python scripts/build_meltome.py
python scripts/embed_esm.py
```

`data/*.parquet` and `cache/` are not checked in. `data/meltome.manifest.json` holds the expected `content_sha256`; if it does not match on your machine, split files do not apply to your parquet.

## Out of scope

- A research question or a contribution to research
- Beating the state of the art
- New architectures
- Protein engineering or mutation prediction
- Scraping: all data comes from a download

If a task starts heading in one of these directions, stop and ask.

## Open points

See the briefing, [section 12](PROJEKTBEESCHREIBUNG.md#12-offene-punkte): lysate only or both measurement types, the identity threshold, whether the CNN stays in, logistic regression or MLP for stage 1, Streamlit or Gradio for the app, and an optional growth-temperature variant of stage 1.
