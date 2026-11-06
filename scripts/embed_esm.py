#!/usr/bin/env python3
"""Compute frozen, mean-pooled ESM-2 embeddings for data/meltome.parquet.

Usage:
    python scripts/embed_esm.py                 # ESM-2 35M
    python scripts/embed_esm.py --model 8M

Requires torch, fair-esm, numpy, pandas and pyarrow.

Embeddings depend only on sequences, so they are computed once per unique sequence
(rule 4) and stay valid when rows are removed from the parquet. Sequences longer than
1022 residues are cut to their first 1022, the ESM-2 limit. A few sequences become
identical through this and then share one embedding row.

Output, in --out-dir (default cache/):
    embeddings.npy         float32, one row per unique truncated sequence
    embeddings_index.parquet  one row per parquet row: id, sequence_sha256, embedding_row
    embeddings.run.json    model, truncation rule, parquet hash, counts, timings

Progress is saved in chunks under --work-dir, outside git and OneDrive. Rerun the same
command after a crash and it continues from the last saved chunk.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MODELS = {"8M": ("esm2_t6_8M_UR50D", 6, 320), "35M": ("esm2_t12_35M_UR50D", 12, 480)}
MAX_RESIDUES = 1022
SEED = 42


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def keep_awake() -> None:
    """Block idle sleep on Windows while this process runs. Closing the lid can still sleep."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def make_batches(lengths: dict[int, int], max_tokens: int, max_seqs: int) -> list[list[int]]:
    """Group sequence numbers, longest first, so padding is low and memory limits show early."""
    batches, current, longest = [], [], 0
    for number in sorted(lengths, key=lambda n: (-lengths[n], n)):
        longest = longest if current else lengths[number]
        if current and ((len(current) + 1) * (longest + 2) > max_tokens or len(current) >= max_seqs):
            batches.append(current)
            current, longest = [], lengths[number]
        current.append(number)
    if current:
        batches.append(current)
    return batches


def load_model(name: str):
    import esm

    original = torch.load
    # fair-esm checkpoints hold argparse objects, which torch >= 2.6 refuses by default.
    torch.load = lambda *a, **k: original(*a, **{**k, "weights_only": False})
    try:
        model, alphabet = getattr(esm.pretrained, name)()
    finally:
        torch.load = original
    return model.eval(), alphabet


def embed_batch(model, converter, layer: int, sequences: list[str]) -> np.ndarray:
    _, _, tokens = converter([(str(i), s) for i, s in enumerate(sequences)])
    with torch.inference_mode():
        reps = model(tokens, repr_layers=[layer])["representations"][layer]
    return torch.stack([reps[i, 1 : len(s) + 1].mean(0) for i, s in enumerate(sequences)]).numpy().astype("float32")


def load_chunks(work_dir: Path, count: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    vectors, done = np.zeros((count, dim), dtype="float32"), np.zeros(count, dtype=bool)
    for path in sorted(work_dir.glob("chunk_*.npz")):
        with np.load(path) as chunk:
            vectors[chunk["rows"]] = chunk["vectors"]
            done[chunk["rows"]] = True
    return vectors, done


def save_chunk(work_dir: Path, number: int, rows: list[int], vectors: np.ndarray) -> None:
    temporary = work_dir / f"partial_{number:06d}.npz"
    np.savez(temporary, rows=np.array(rows), vectors=vectors)
    os.replace(temporary, work_dir / f"chunk_{number:06d}.npz")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, default="35M")
    parser.add_argument("--parquet", type=Path, default=root / "data/meltome.parquet")
    parser.add_argument("--out-dir", type=Path, default=root / "cache")
    parser.add_argument("--work-dir", type=Path, help="Default: ~/.cache/dsf-asnv/embeddings/<model>")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-seqs", type=int, default=16)
    parser.add_argument("--chunk-batches", type=int, default=100)
    parser.add_argument("--limit", type=int, help="Use a random subset of this many sequences, for testing.")
    args = parser.parse_args()

    name, layer, dim = MODELS[args.model]
    work_dir = args.work_dir or Path.home() / ".cache/dsf-asnv/embeddings" / args.model
    work_dir.mkdir(parents=True, exist_ok=True)
    keep_awake()
    torch.manual_seed(SEED)

    table = pd.read_parquet(args.parquet)
    truncated = table["sequence"].str.slice(0, MAX_RESIDUES)
    unique = sorted(truncated.unique().tolist())
    if args.limit:
        unique = sorted(random.Random(SEED).sample(unique, args.limit))
    row_of = {sequence: row for row, sequence in enumerate(unique)}

    config = {
        "model": name,
        "layer": layer,
        "dim": dim,
        "max_residues": MAX_RESIDUES,
        "pooling": "mean over residues, excluding BOS and EOS",
        "sequences_sha256": sha256_text("\n".join(unique)),
    }
    config_path = work_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit(f"{work_dir} holds chunks from a different run. Use another --work-dir or empty it.")
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    vectors, done = load_chunks(work_dir, len(unique), dim)
    pending = {i: len(s) for i, s in enumerate(unique) if not done[i]}
    print(f"{args.model}: {len(unique)} unique sequences, {int(done.sum())} already saved, {len(pending)} to go", flush=True)

    started = time.perf_counter()
    if pending:
        model, alphabet = load_model(name)
        converter = alphabet.get_batch_converter()
        batches = make_batches(pending, args.max_tokens, args.max_seqs)
        total, finished = sum(pending.values()), 0
        chunk_number = len(list(work_dir.glob("chunk_*.npz")))
        rows, parts = [], []
        for index, batch in enumerate(batches, 1):
            output = embed_batch(model, converter, layer, [unique[i] for i in batch])
            if not np.isfinite(output).all():
                raise SystemExit(f"Non-finite embedding in batch {index}.")
            rows += batch
            parts.append(output)
            finished += sum(pending[i] for i in batch)
            if index % args.chunk_batches == 0 or index == len(batches):
                save_chunk(work_dir, chunk_number, rows, np.vstack(parts))
                chunk_number += 1
                rows, parts = [], []
                elapsed = time.perf_counter() - started
                rate = finished / elapsed
                print(
                    f"{time.strftime('%H:%M:%S')} batch {index}/{len(batches)}, {finished / total:.1%} of residues, "
                    f"{rate:.0f} residues/s, about {(total - finished) / rate / 3600:.1f} h left",
                    flush=True,
                )
        vectors, done = load_chunks(work_dir, len(unique), dim)

    if not done.all() or not np.isfinite(vectors).all() or (vectors == 0).all(axis=1).any():
        raise SystemExit("Embedding matrix is incomplete. Rerun to continue.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "embeddings.npy", vectors)
    index = pd.DataFrame(
        {
            "id": table["id"],
            "sequence_sha256": table["sequence"].map(sha256_text),
            "embedding_row": truncated.map(row_of),
        }
    ).dropna(subset=["embedding_row"]).astype({"embedding_row": "int64"})
    index.to_parquet(args.out_dir / "embeddings_index.parquet", index=False)
    record = {
        **config,
        "parquet_sha256": sha256_file(args.parquet),
        "parquet_rows": len(table),
        "indexed_rows": len(index),
        "embedding_rows": len(unique),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "seed": SEED,
        "seconds_this_session": round(time.perf_counter() - started, 1),
    }
    (args.out_dir / "embeddings.run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"Done. Wrote {vectors.shape} to {args.out_dir / 'embeddings.npy'}")


if __name__ == "__main__":
    main()
