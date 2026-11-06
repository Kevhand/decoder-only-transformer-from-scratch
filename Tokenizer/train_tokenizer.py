"""
Train the byte-level BPE tokenizer on a sample of FineWeb-Edu (or local text files).

Examples:
    # Stream ~300 MB of FineWeb-Edu from Hugging Face (needs: pip install datasets)
    python train_tokenizer.py --vocab-size 32768 --sample-mb 300 --out tokenizer.json

    # Or train on local .txt / .jsonl files (jsonl lines need a "text" field)
    python train_tokenizer.py --files data/*.txt --vocab-size 32768 --out tokenizer.json

Pre-tokenizing is the slow, CPU-bound step (~1 MB/s per core in pure Python),
so it runs in parallel worker processes. The merge step runs on one core.
"""

import argparse
import json
import os
import time
from collections import Counter
from multiprocessing import Pool

from byte_level_bpe import ByteLevelBPETokenizer, count_chunks


# ------------------------------------------------------------
# Reading documents
# ------------------------------------------------------------

def iter_hf_documents(dataset, name, split):
    from datasets import load_dataset  # imported here so local-file use needs no extra package

    stream = load_dataset(dataset, name=name, split=split, streaming=True)
    for row in stream:
        yield row["text"]


def iter_local_documents(paths):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)["text"]
            else:
                # Blank lines separate documents in plain text files.
                for doc in f.read().split("\n\n"):
                    if doc.strip():
                        yield doc


def take_documents(documents, sample_bytes, holdout_docs):
    """Split the stream into a training sample of about sample_bytes and a small held-out set."""
    train, holdout = [], []
    total = 0

    for doc in documents:
        if len(holdout) < holdout_docs:
            holdout.append(doc)
            continue

        train.append(doc)
        total += len(doc.encode("utf-8"))
        if total >= sample_bytes:
            break

    return train, holdout, total


def batched(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=32768,
                        help="Target vocab size. Keep it <= 65536 so token ids fit in uint16.")
    parser.add_argument("--sample-mb", type=float, default=300,
                        help="How much text to train on, in MB.")
    parser.add_argument("--files", nargs="*", help="Local .txt or .jsonl files instead of Hugging Face.")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-name", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--holdout-docs", type=int, default=500)
    parser.add_argument("--out", default="tokenizer.json")
    args = parser.parse_args()

    if args.vocab_size > 65536:
        raise ValueError("Use a vocab size <= 65536 so the tokenized dataset can be stored as uint16.")

    # 1. Collect the training sample (the held-out docs come first and are never trained on).
    t0 = time.time()
    if args.files:
        documents = iter_local_documents(args.files)
    else:
        documents = iter_hf_documents(args.dataset, args.dataset_name, args.split)

    train_docs, holdout_docs, total_bytes = take_documents(
        documents, int(args.sample_mb * 1e6), args.holdout_docs
    )
    print(f"Collected {len(train_docs):,} documents ({total_bytes / 1e6:.1f} MB) "
          f"+ {len(holdout_docs)} held-out in {time.time() - t0:.0f}s")

    # 2. Pre-tokenize and count chunks in parallel.
    t0 = time.time()
    chunk_counts = Counter()
    with Pool(args.workers) as pool:
        for partial in pool.imap_unordered(count_chunks, batched(train_docs, 1000)):
            chunk_counts.update(partial)
    print(f"Pre-tokenized with {args.workers} workers in {time.time() - t0:.0f}s: "
          f"{len(chunk_counts):,} distinct chunks")

    del train_docs  # free memory before merging

    # 3. Learn merges.
    t0 = time.time()
    tokenizer = ByteLevelBPETokenizer(target_vocab_size=args.vocab_size)
    tokenizer.train_from_counts(chunk_counts, verbose=2000)
    print(f"Learned {len(tokenizer.merges):,} merges in {time.time() - t0:.0f}s; "
          f"vocab size {tokenizer.vocab_size:,}")

    tokenizer.save(args.out)
    print(f"Saved to {args.out}")

    # 4. Check on held-out documents the tokenizer never saw.
    loaded = ByteLevelBPETokenizer.load(args.out)
    n_bytes = n_tokens = 0
    for doc in holdout_docs:
        ids = loaded.encode(doc)
        assert loaded.decode(ids) == doc, "Round trip failed on a held-out document"
        assert loaded.unk_id not in ids
        n_bytes += len(doc.encode("utf-8"))
        n_tokens += len(ids)

    print(f"Held-out: {len(holdout_docs)} docs round-trip exactly; "
          f"{n_bytes / n_tokens:.2f} bytes/token")

    sample = holdout_docs[0][:200]
    pieces = [loaded.decode([i]) for i in loaded.encode(sample)]
    print("Example split:", pieces[:40])


if __name__ == "__main__":
    main()
