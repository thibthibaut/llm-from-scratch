# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub[hf_xet]", "pyarrow"]
# ///
"""Build the fineweb-edu SQLite database that `train` reads, without burn's Python importer.

burn's `HuggingfaceDatasetLoader` can build this file itself, but it goes through
`datasets.load_dataset`, which keeps the downloaded parquet *and* an Arrow cache next to the
final `.db` — roughly twice the disk the database alone needs. This script instead downloads one
parquet shard at a time, appends its `text` column to the database, and deletes the shard before
fetching the next, so peak disk usage is the database plus a single ~2 GB shard.

The output matches what `SqliteDataset::from_db_file(path, "train")` expects: a `train` table
whose `row_id` primary key starts at 1 and has no gaps, plus the `text` column that `TextItem`
deserializes. Only `text` is kept; fineweb-edu's metadata columns are never read by the trainer.

Run it with `uv`, which resolves the inline dependencies above:

    uv run scripts/build_fineweb_db.py --output /mnt/data/fineweb-edu/fineweb-edu-sample-10BT.db

Then point the trainer at it with `FINEWEB_DB=<that path>` (see `src/dataset.rs`).

The script is resumable: shards already recorded in the `_shards` table are skipped, and a shard
that was interrupted half-way is rolled back, because each shard is inserted in one transaction.
"""

import argparse
import sqlite3
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "HuggingFaceFW/fineweb-edu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", required=True, type=Path, help="SQLite file to create")
    parser.add_argument(
        "--subset",
        default="sample/10BT",
        help="Directory inside the HF repo to import (default: sample/10BT)",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="Where to stage each parquet shard (default: next to --output, never the root disk)",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scratch_root = args.scratch_dir or args.output.parent
    scratch_root.mkdir(parents=True, exist_ok=True)

    shards = sorted(
        entry.path
        for entry in HfApi().list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=args.subset)
        if entry.path.endswith(".parquet")
    )
    if not shards:
        raise SystemExit(f"no parquet shards under {REPO_ID}/{args.subset}")
    print(f"{len(shards)} shards in {REPO_ID}/{args.subset}")

    db = sqlite3.connect(args.output)
    # Bulk load: durability is irrelevant (an interrupted shard is rolled back and redone), and
    # the default journal would double the write volume.
    db.execute("PRAGMA journal_mode = OFF")
    db.execute("PRAGMA synchronous = OFF")
    db.execute("CREATE TABLE IF NOT EXISTS train (row_id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS _shards (path TEXT PRIMARY KEY, rows INTEGER)")
    db.commit()
    done = {row[0] for row in db.execute("SELECT path FROM _shards")}

    for i, shard in enumerate(shards, 1):
        if shard in done:
            print(f"[{i}/{len(shards)}] {shard}: already imported, skipping")
            continue

        with tempfile.TemporaryDirectory(dir=scratch_root, prefix="shard-") as tmp:
            print(f"[{i}/{len(shards)}] downloading {shard}", flush=True)
            local = hf_hub_download(REPO_ID, shard, repo_type="dataset", local_dir=tmp)

            rows = 0
            with db:  # one transaction per shard: all of it lands, or none of it does
                for batch in pq.ParquetFile(local).iter_batches(columns=["text"], batch_size=8192):
                    texts = batch.column(0).to_pylist()
                    # row_id is left NULL so SQLite assigns max(row_id) + 1: contiguous from 1,
                    # which is the invariant SqliteDataset's `get(index)` relies on.
                    db.executemany("INSERT INTO train (text) VALUES (?)", ((t,) for t in texts))
                    rows += len(texts)
                db.execute("INSERT INTO _shards (path, rows) VALUES (?, ?)", (shard, rows))
            print(f"[{i}/{len(shards)}] {shard}: {rows} rows", flush=True)

    total = db.execute("SELECT coalesce(max(row_id), 0) FROM train").fetchone()[0]
    db.close()
    print(f"done: {total} rows in {args.output}")


if __name__ == "__main__":
    main()
