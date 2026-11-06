# Tm Prediction and Organism Leakage

How much of the predictive performance for protein melting temperatures (Tm) comes from real sequence biophysics, and how much from the organism a protein originates from?

**Timeline:** 2026-11-09 to 2026-11-27, team of four.

This README is an overview. The full project briefing, in German, is [PROJEKTBEESCHREIBUNG.md](PROJEKTBEESCHREIBUNG.md). It is the single source of truth: read it before taking on a task, and agree on deviations beforehand.

## Background

The Meltome Atlas contains the proteomes of 13 organisms. Thermophilic organisms have consistently more stable proteomes, so a model that recognizes a sequence's lineage can predict Tm respectably without knowing anything about stability. This is leakage at the organism level, on top of the usual homology leakage.

## Hypotheses

| ID | Hypothesis | Tested by |
| --- | --- | --- |
| H1 | A random split substantially overestimates generalization | Condition 1, drop across identity thresholds |
| H2 | A large part of the performance is explained by organism identity alone | Organism median baseline vs. models; condition 2 |
| H3 | Within a single proteome, a reduced but measurable signal remains | Condition 3 |
| H4 | A model trained from scratch exploits the origin shortcut more than a pretrained one | CNN vs. ESM-2 across conditions 0, 2, 3 |

H3 and H4 are open questions, and answering them is the project's contribution. A negative result is still a result: if no model beats the organism baseline, that is the project's finding, not its failure.

## Experiment design

Four evaluation conditions, nested from permissive to strict. The data, pipeline and metrics stay the same; only the split changes.

| # | Condition | Removes | Folds / repetitions |
| --- | --- | --- | --- |
| 0 | Random split | Nothing | 5 seeds |
| 1 | Identity clusters at 90 / 70 / 50 / 30 % | Close homologs | 5 cluster assignments per threshold |
| 2 | Leave-one-organism-out | Organism signal | 13 folds |
| 3 | Within one organism | Holds organism signal constant | 5-fold CV per suitable organism |

### Models

| Name | Model | Source of knowledge | Conditions |
| --- | --- | --- | --- |
| `comp` | 20 amino acid frequencies + length, Ridge | Statistics only | 0, 1, 2, 3 |
| `esm` | ESM-2 (8M or 35M), mean-pooled, frozen + Ridge | Pretraining | 0, 1, 2, 3 |
| `cnn` | 1D CNN on one-hot encoding, from scratch | Training set only | 0, 2, 3 |

`cnn` deliberately skips condition 1. The full identity curve is reserved for `esm` to keep week 2 feasible.

### Baselines

| Name | Prediction | Purpose |
| --- | --- | --- |
| `mean` | Training set mean | Lower bound |
| `organism` | Median Tm of the source organism, ignoring the sequence | Tests H2 |

`organism` is undefined on condition 2 and constant on condition 3. This is intended, and the results table records `null` there.

### Metrics

The primary metric is **RMSE in °C**, the secondary metric is **Spearman**.

Every result row must also include `test_std`, the standard deviation of Tm in that test set. Without it, RMSE is not comparable across conditions: the Tm range within one proteome is smaller, so RMSE drops automatically even if the model has learned nothing.

## Interfaces

These formats are fixed at the end of day 2 and only change afterwards if everyone agrees.

### Dataset: `data/meltome.parquet`

| Column | Type | Description |
| --- | --- | --- |
| `id` | str | UniProt accession, unique |
| `sequence` | str | Amino acid sequence, uppercase |
| `tm` | float | Melting temperature in °C |
| `organism` | str | Source organism, normalized |
| `condition` | str | `lysate` or `cell` |

### Splits: `splits/{condition}_{parameter}_{repetition}.json`

For example, `cond1_id30_rep3.json`:

```json
{
  "condition": 1,
  "params": {"identity": 0.30},
  "rep": 3,
  "train": [0, 4, 7, 11],
  "test": [1, 2, 9]
}
```

Indices refer to row numbers in `data/meltome.parquet`. Split files are checked in, not regenerated.

### Evaluation

```python
def evaluate(train_idx, test_idx, model_name) -> dict:
    """Returns: {rmse, spearman, mae, n_train, n_test, test_std}"""
```

Results are stored in `results/results.csv`, one row per split and model.

## Hard rules

1. **Splits belong to role A.** Nobody else creates splits. Model code loads split files and never computes them.
2. **No hyperparameter selection on the test set.** Every search runs in an inner CV within the training set.
3. **No fine-tuning of ESM-2.** It does not answer the research question.
4. **Embeddings are computed once and cached**, not recomputed per split.
5. **Nothing new after Gate 2 (2026-11-20).** Week 3 is for analysis and writing.
6. **Every result is reproducible:** fixed seed, recorded in the result row.

## Roles

| Role | Responsibility | Delivers |
| --- | --- | --- |
| A | Data, organism labels, MMseqs2 clustering, all splits | `data/meltome.parquet`, `splits/*.json` |
| B | ESM-2 embeddings, Ridge, CNN training | `cache/embeddings.npy`, `train.py` |
| C | Metrics, baselines, evaluation loop | `evaluate.py`, `results/results.csv` |
| D | Figures, noise estimate, interpretation, paper | `figures/`, `paper/` |

D supports A and C in weeks 1 and 2, then works independently from Gate 2 onwards.

## Schedule

| Period | A | B | C | D |
| --- | --- | --- | --- | --- |
| Nov 9–13 | Data, check labels | Embeddings, Ridge | Harness, baselines | Supports A and C |
| Nov 16–20 | MMseqs2, all splits | Runs across all conditions | Noise, ablation | Supports C |
| Nov 23–27 | Verify reproduction | Verify reproduction | Verify reproduction | Figures, text |

- **Gate 1, Fri Nov 13:** A number exists on condition 0 for all three models and both baselines. If not, scope is cut to conditions 0 and 2, and the identity curve is dropped.
- **Gate 2, Fri Nov 20:** All conditions have run. Feature freeze from here on.
- **Submission, Fri Nov 27.**

## Out of scope

- Beating the state of the art on a benchmark
- New architectures
- Protein engineering or mutation prediction
- Scraping: all data comes from a download

If a task starts heading in one of these directions, stop and ask.

## Risks and open points

The briefing also covers [risks and fallbacks](PROJEKTBEESCHREIBUNG.md#7-risiken), including how to handle measurement noise, and [open points](PROJEKTBEESCHREIBUNG.md#9-offene-punkte), such as which ESM-2 variant to use and which organisms have enough proteins for condition 3.
