#!/usr/bin/env python3
"""Build data/meltome.parquet (briefing section 3.1) from the FLIP Meltome dump.

Usage:
    python scripts/build_meltome.py
    python scripts/build_meltome.py --both-conditions   # see "One row per id" below

Requires pandas and pyarrow. Two inputs, both pinned by SHA-256:
  - `full_dataset.json.zip` from the FLIP benchmark (J-SNACKKB/FLIP, AFL-3.0), a
    re-export of the Meltome Atlas (Jarzab et al., Nat Methods 2020,
    doi:10.1038/s41592-020-0801-4). Pinned by commit as well.
  - `human_sequences.tsv` from the FLIP data server, a UniProt export of reviewed
    human entries, used only to map human gene symbols to accessions.
Downloads go to ~/.cache/dsf-asnv/meltome-raw, outside git and OneDrive; override
with --cache-dir.

Row numbers in the parquet are the index space of splits/*.json, so the output
order is deterministic. `content_sha256` in data/meltome.manifest.json must match
on every machine; a mismatch means splits do not apply to that parquet.

Labels: every `runName` maps to organism and condition in RUNS. The dump only
labels this for the non-human runs. The human runs were matched to Supplementary
Table 1 of the paper by replicate count, temperature window and number of fitted
curves: all ten are `cells` experiments.

Human accessions: human rows are keyed by gene symbol, and the accession in the
dump is unreliable because it looks substring-matched (ATP6 became SLCO6A1 through
its synonym OATP6A1). So each symbol is resolved again by exact match: the entry
whose primary gene name it is, otherwise the entry that lists it as a synonym. A
symbol matching several entries at the deciding step, or none, is dropped.
Sequences for human rows come from the same file.

Filters, in order:
  1. Drop rows without accession, sequence or Tm.
  2. Drop accessions that occur in more than one organism.

One row per id: E. coli and T. thermophilus were measured in both conditions. By
default only their lysate runs are kept, so `id` stays unique as section 3.1
requires. --both-conditions keeps both runs and makes (id, condition) the key.

Tm is `meltingPoint`, the median over all remaining rows of a protein (replicates,
human cell lines, mouse tissues).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

FLIP_COMMIT = "62cace8735f5610e2743cf06ce0f944b37fffaa6"
RAW_NAME = "full_dataset.json.zip"
RAW_URL = f"https://github.com/J-SNACKKB/FLIP/raw/{FLIP_COMMIT}/splits/meltome/{RAW_NAME}"
RAW_SHA256 = "9e43778730760180581493d05b08c493ad897e4390763cfadad092e62aaea05d"
HUMAN_NAME = "human_sequences.tsv"
# Plain HTTP: the server's HTTPS certificate does not match its name. SHA-256 protects the content.
HUMAN_URL = f"http://data.bioembeddings.com/public/FLIP/meltome/{HUMAN_NAME}"
HUMAN_SHA256 = "8d4b71e18db5bf7ecbad3d6d17cdea127ef7d47a939ab787b1b18f34c8c5292c"

HUMAN = "Homo sapiens"
ESM2_MAX_RESIDUES = 1022
COLUMNS = ["id", "sequence", "tm", "organism", "condition"]

RUNS = {
    "Arabidopsis thaliana seedling lysate": ("Arabidopsis thaliana", "lysate"),
    "Bacillus subtilis_168_lysate_R1": ("Bacillus subtilis", "lysate"),
    "Caenorhabditis elegans lysate": ("Caenorhabditis elegans", "lysate"),
    "Danio rerio Zenodo lysate": ("Danio rerio", "lysate"),
    "Drosophila melanogaster SII lysate": ("Drosophila melanogaster", "lysate"),
    "Escherichia coli cells": ("Escherichia coli", "cell"),
    "Escherichia coli lysate": ("Escherichia coli", "lysate"),
    "Geobacillus stearothermophilus NCA26 lysate": ("Geobacillus stearothermophilus", "lysate"),
    "Mus musculus BMDC lysate": ("Mus musculus", "lysate"),
    "Mus musculus liver lysate": ("Mus musculus", "lysate"),
    "Oleispira antarctica_RB-8_lysate_R1": ("Oleispira antarctica", "lysate"),
    "Picrophilus torridus DSM9790 lysate": ("Picrophilus torridus", "lysate"),
    "Saccharomyces cerevisiae lysate": ("Saccharomyces cerevisiae", "lysate"),
    "Thermus thermophilus HB27 cells": ("Thermus thermophilus", "cell"),
    "Thermus thermophilus HB27 lysate": ("Thermus thermophilus", "lysate"),
    "HAOEC": (HUMAN, "cell"),
    "HEK293T": (HUMAN, "cell"),
    "HL60": (HUMAN, "cell"),
    "HaCaT": (HUMAN, "cell"),
    "HepG2": (HUMAN, "cell"),
    "Jurkat": (HUMAN, "cell"),
    "K562": (HUMAN, "cell"),
    "U937": (HUMAN, "cell"),
    "colon_cancer_spheroids": (HUMAN, "cell"),
    "pTcells": (HUMAN, "cell"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, expected_sha256: str, cache_dir: Path) -> Path:
    """Return the verified file, downloading it if it is not cached."""
    path = cache_dir / url.rsplit("/", 1)[1]
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        partial = path.with_name(path.name + ".part")
        # The FLIP data server answers 406 to Python's default User-Agent.
        request = urllib.request.Request(url, headers={"User-Agent": "dsf-asnv-build-meltome"})
        with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        partial.replace(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise SystemExit(f"SHA-256 mismatch for {path}: {actual}. Delete it and rerun.")
    return path


def load_raw(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive, archive.open("full_dataset.json") as handle:
        records = json.load(handle)
    frame = pd.DataFrame(
        {
            "gene": [r["proteinId"] for r in records],
            "id": [r.get("uniprotAccession") for r in records],
            "run": [r["runName"] for r in records],
            "tm": [r["meltingPoint"] for r in records],
            "sequence": [r.get("sequence") for r in records],
        }
    )
    unknown = set(frame["run"]) - set(RUNS)
    if unknown:
        raise SystemExit(f"Unmapped runName values: {sorted(unknown)}")
    frame["organism"] = frame["run"].map(lambda run: RUNS[run][0])
    frame["condition"] = frame["run"].map(lambda run: RUNS[run][1])
    return frame


def remap_human(raw: pd.DataFrame, uniprot_path: Path) -> tuple[pd.DataFrame, dict]:
    """Replace accession and sequence of human rows by an exact gene-symbol match."""
    primary_column = "Gene names  (primary )"
    uniprot = pd.read_csv(
        uniprot_path, sep="\t", usecols=["Entry", "Gene names", primary_column, "Sequence"], keep_default_na=False
    )
    primary: dict[str, set[str]] = {}
    synonym: dict[str, set[str]] = {}
    for entry, names, primaries in zip(uniprot["Entry"], uniprot["Gene names"], uniprot[primary_column]):
        for symbol in filter(None, re.split(r"[;\s]+", primaries)):
            primary.setdefault(symbol, set()).add(entry)
        for symbol in filter(None, re.split(r"[;\s]+", names)):
            synonym.setdefault(symbol, set()).add(entry)

    is_human = raw["organism"].eq(HUMAN)
    dump_ids = raw.loc[is_human].drop_duplicates("gene").set_index("gene")["id"]
    mapping, outcome = {}, {"primary": 0, "synonym": 0, "ambiguous": 0, "unknown": 0}
    for symbol in dump_ids.index:
        candidates = primary.get(symbol) or synonym.get(symbol)
        if not candidates:
            outcome["unknown"] += 1
        elif len(candidates) > 1:
            outcome["ambiguous"] += 1
        else:
            outcome["primary" if symbol in primary else "synonym"] += 1
            mapping[symbol] = next(iter(candidates))

    remapped = raw.copy()
    remapped.loc[is_human, "id"] = remapped.loc[is_human, "gene"].map(mapping)
    remapped.loc[is_human, "sequence"] = remapped.loc[is_human, "id"].map(uniprot.set_index("Entry")["Sequence"])
    new_ids = pd.Series(mapping)
    changed = dump_ids.reindex(new_ids.index)
    log = {
        "human_symbols_resolved_by": outcome,
        "human_symbols_with_new_accession": int((changed.notna() & changed.ne(new_ids)).sum()),
        "human_symbols_without_accession_in_dump_now_resolved": int(changed.isna().sum()),
    }
    return remapped, log


def build(raw: pd.DataFrame, both_conditions: bool) -> tuple[pd.DataFrame, dict]:
    log = {"raw_rows": len(raw)}

    usable = raw.dropna(subset=["id", "sequence", "tm"])
    log["dropped_rows_missing_id_sequence_or_tm"] = len(raw) - len(usable)

    organisms = usable.groupby("id")["organism"].nunique()
    shared = set(organisms[organisms > 1].index)
    log["dropped_ids_in_multiple_organisms"] = sorted(shared)
    usable = usable[~usable["id"].isin(shared)]

    has_both = usable.groupby(["organism", "id"])["condition"].nunique().eq(2)
    log["proteins_measured_in_both_conditions"] = (
        has_both[has_both].groupby("organism").size().to_dict()
    )
    if not both_conditions:
        dual = usable.groupby("organism")["condition"].nunique()
        dual = dual[dual > 1].index
        mask = usable["organism"].isin(dual) & usable["condition"].eq("cell")
        log["dropped_cell_rows_for_dual_condition_organisms"] = int(mask.sum())
        usable = usable[~mask]

    if usable.groupby("id")["sequence"].nunique().max() != 1:
        raise SystemExit("An id maps to more than one sequence.")

    table = (
        usable.groupby(["id", "condition"], as_index=False)
        .agg(sequence=("sequence", "first"), tm=("tm", "median"), organism=("organism", "first"))
        .sort_values(["id", "condition"], ignore_index=True)[COLUMNS]
    )
    table["tm"] = table["tm"].astype("float64")
    for column in ("id", "sequence", "organism", "condition"):
        table[column] = table[column].astype("string[pyarrow]")
    return table, log


def validate(table: pd.DataFrame, both_conditions: bool) -> None:
    key = ["id", "condition"] if both_conditions else ["id"]
    checks = {
        "columns": list(table.columns) == COLUMNS,
        "no_nulls": not table.isna().any().any(),
        "key_unique": not table.duplicated(key).any(),
        "uppercase": table["sequence"].str.fullmatch("[A-Z]+").all(),
        "condition_values": set(table["condition"]) <= {"lysate", "cell"},
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"Validation failed: {failed}")


def content_hash(table: pd.DataFrame) -> str:
    """Hash of row content and order, independent of parquet encoding."""
    digest = hashlib.sha256()
    for row in table.itertuples(index=False):
        line = f"{row.id}\t{row.sequence}\t{row.tm:.6f}\t{row.organism}\t{row.condition}\n"
        digest.update(line.encode())
    return digest.hexdigest()


def replicate_summary(raw: pd.DataFrame) -> dict:
    """Spread of repeated Tm values per gene symbol within one human run.

    The repeats carry no replicate label, so only their spread can be computed.
    """
    rows = raw[raw["organism"].eq(HUMAN) & raw["tm"].notna()]
    groups = rows.groupby(["run", "gene"])["tm"].agg(["size", "std"])
    by_run = groups.groupby("run").agg(median_repeats=("size", "median"), median_sd=("std", "median"))
    return {run: {k: round(float(v), 3) for k, v in row.items()} for run, row in by_run.iterrows()}


def condition_agreement(raw: pd.DataFrame, organism: str, left: str, right: str) -> dict:
    subset = raw[raw["organism"].eq(organism)].dropna(subset=["id", "tm"])
    a = subset[subset["run"].eq(left)].set_index("id")["tm"]
    b = subset[subset["run"].eq(right)].set_index("id")["tm"]
    shared = a.index.intersection(b.index)
    delta = a[shared] - b[shared]
    return {
        "shared_proteins": len(shared),
        "pearson": round(float(a[shared].corr(b[shared])), 3),
        "rmse_c": round(float((delta**2).mean() ** 0.5), 2),
    }


def summarize(raw: pd.DataFrame, table: pd.DataFrame, log: dict) -> dict:
    per_organism = table.groupby(["organism", "condition"]).agg(
        rows=("id", "size"),
        tm_median=("tm", "median"),
        over_esm2_limit=("sequence", lambda s: int((s.str.len() > ESM2_MAX_RESIDUES).sum())),
    )
    lengths = table["sequence"].str.len()
    sequences = table.groupby("sequence")["organism"].agg(["size", "nunique"])
    return {
        "sources": [{"url": RAW_URL, "sha256": RAW_SHA256}, {"url": HUMAN_URL, "sha256": HUMAN_SHA256}],
        "rows": len(table),
        "unique_ids": int(table["id"].nunique()),
        "unique_sequences": int(table["sequence"].nunique()),
        "content_sha256": content_hash(table),
        "condition_rows": table["condition"].value_counts().sort_index().to_dict(),
        "per_organism": [
            {
                "organism": organism,
                "condition": condition,
                "rows": int(row["rows"]),
                "tm_median": round(float(row["tm_median"]), 2),
                "over_esm2_limit": int(row["over_esm2_limit"]),
            }
            for (organism, condition), row in per_organism.iterrows()
        ],
        "identical_sequences": {
            "sequences_on_multiple_rows": int((sequences["size"] > 1).sum()),
            "rows_affected": int(sequences.loc[sequences["size"] > 1, "size"].sum()),
            "sequences_across_organisms": int((sequences["nunique"] > 1).sum()),
        },
        f"rows_over_{ESM2_MAX_RESIDUES}_residues": int((lengths > ESM2_MAX_RESIDUES).sum()),
        "longest_sequence": int(lengths.max()),
        "rows_with_X_or_U": int(table["sequence"].str.contains("[XU]").sum()),
        "filters": log,
        "human_replicates_by_run": replicate_summary(raw),
        "agreement_between_runs": {
            "E. coli cell vs lysate": condition_agreement(
                raw, "Escherichia coli", "Escherichia coli cells", "Escherichia coli lysate"
            ),
            "T. thermophilus cell vs lysate": condition_agreement(
                raw, "Thermus thermophilus", "Thermus thermophilus HB27 cells", "Thermus thermophilus HB27 lysate"
            ),
            "M. musculus BMDC vs liver": condition_agreement(
                raw, "Mus musculus", "Mus musculus BMDC lysate", "Mus musculus liver lysate"
            ),
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/dsf-asnv/meltome-raw")
    parser.add_argument("--out", type=Path, default=root / "data/meltome.parquet")
    parser.add_argument("--both-conditions", action="store_true", help="Keep cell and lysate rows; id is no longer unique.")
    args = parser.parse_args()

    raw = load_raw(fetch(RAW_URL, RAW_SHA256, args.cache_dir))
    raw, mapping_log = remap_human(raw, fetch(HUMAN_URL, HUMAN_SHA256, args.cache_dir))
    table, log = build(raw, args.both_conditions)
    log = {**mapping_log, **log}
    validate(table, args.both_conditions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    summary = summarize(raw, table, log)
    manifest = args.out.with_name(args.out.stem + ".manifest.json")
    manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) and {manifest}")


if __name__ == "__main__":
    main()
