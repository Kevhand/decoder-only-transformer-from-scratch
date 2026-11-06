"""
Tokenize FineWeb-Edu into uint16 shards for training.

Each document becomes [BOS] + tokens + [EOS], and documents are packed back to
back with no padding. Tokens are written into fixed-size shards saved as .npy
files that the training loop can memory-map.

The first --val-tokens tokens go to a validation shard; everything after that
goes to training shards.

    pip install datasets numpy
    python tokenize_dataset.py --tokenizer tokenizer.json --out-dir data --max-tokens 2.5e9

To extend the dataset later, rerun with --resume: it reads data/manifest.json
and continues from the first document that wasn't tokenized.

Local files (.txt or .jsonl with a "text" field) can be used instead of
streaming with --files.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from byte_level_bpe import ByteLevelBPETokenizer
from train_tokenizer import iter_hf_documents, iter_local_documents


# ------------------------------------------------------------
# Worker processes: each loads the tokenizer once and keeps its own cache
# ------------------------------------------------------------

_tokenizer = None


def init_worker(tokenizer_path):
    global _tokenizer
    _tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)


def encode_batch(docs):
    bos, eos = _tokenizer.bos_id, _tokenizer.eos_id
    ids = []
    n_bytes = 0
    for doc in docs:
        ids.append(bos)
        ids.extend(_tokenizer.encode(doc))
        ids.append(eos)
        n_bytes += len(doc.encode("utf-8"))
    return np.array(ids, dtype=np.uint16), len(docs), n_bytes


def batched(documents, batch_size):
    batch = []
    for doc in documents:
        batch.append(doc)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ------------------------------------------------------------
# Shard writer
# ------------------------------------------------------------

class ShardWriter:
    """Fills fixed-size uint16 buffers and saves each one as a .npy shard."""

    def __init__(self, out_dir, shard_size, next_index):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.index = next_index
        self.buffer = np.empty(shard_size, dtype=np.uint16)
        self.filled = 0
        self.files = []

    def write(self, tokens, split):
        while len(tokens) > 0:
            space = self.shard_size - self.filled
            take = min(space, len(tokens))
            self.buffer[self.filled:self.filled + take] = tokens[:take]
            self.filled += take
            tokens = tokens[take:]
            if self.filled == self.shard_size:
                self.flush(split)

    def flush(self, split):
        if self.filled == 0:
            return
        path = os.path.join(self.out_dir, f"fineweb_edu_{split}_{self.index:06d}.npy")
        np.save(path, self.buffer[:self.filled])
        self.files.append(path)
        print(f"  wrote {os.path.basename(path)} ({self.filled:,} tokens)")
        self.index += 1
        self.filled = 0


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--max-tokens", type=float, default=2.5e9,
                        help="Stop after about this many training tokens (validation not included).")
    parser.add_argument("--val-tokens", type=float, default=10e6)
    parser.add_argument("--shard-size", type=float, default=100e6, help="Tokens per shard (100M = 200 MB).")
    parser.add_argument("--files", nargs="*", help="Local .txt or .jsonl files instead of streaming.")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-name", default="sample-10BT")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--batch-docs", type=int, default=256)
    parser.add_argument("--resume", action="store_true", help="Continue from data/manifest.json.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.json")

    tokenizer = ByteLevelBPETokenizer.load(args.tokenizer)
    if tokenizer.vocab_size > 65536:
        raise ValueError("Vocab is too large for uint16 token ids.")

    # Where to start: fresh, or after the documents a previous run already used.
    if args.resume:
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest["tokenizer_merges"] != len(tokenizer.merges):
            raise ValueError("This tokenizer differs from the one used for the existing shards.")
        skip_docs = manifest["docs_consumed"]
        val_done = True
        train_tokens = manifest["train_tokens"]
        next_shard = manifest["next_shard"]
        print(f"Resuming after {skip_docs:,} documents ({train_tokens:,} training tokens so far)")
    else:
        manifest = {"tokenizer": os.path.abspath(args.tokenizer),
                    "tokenizer_merges": len(tokenizer.merges),
                    "vocab_size": tokenizer.vocab_size,
                    "bos_id": tokenizer.bos_id, "eos_id": tokenizer.eos_id, "pad_id": tokenizer.pad_id,
                    "val_files": [], "train_files": [], "val_tokens": 0,
                    "dataset": args.files or f"{args.dataset}/{args.dataset_name}"}
        skip_docs, val_done, train_tokens, next_shard = 0, False, 0, 0

    documents = iter_local_documents(args.files) if args.files else iter_hf_documents(
        args.dataset, args.dataset_name, "train")

    def skipped(docs, n):
        for i, doc in enumerate(docs):
            if i >= n:
                yield doc

    documents = skipped(documents, skip_docs)

    writer = ShardWriter(args.out_dir, int(args.shard_size), next_shard)
    val_target = int(args.val_tokens)
    train_target = int(args.max_tokens)
    val_tokens = manifest["val_tokens"]
    docs_consumed = skip_docs
    total_bytes = 0
    start = time.time()
    last_report = start

    with Pool(args.workers, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
        # imap keeps batches in document order, so runs are reproducible.
        for tokens, n_docs, n_bytes in pool.imap(encode_batch, batched(documents, args.batch_docs)):
            docs_consumed += n_docs
            total_bytes += n_bytes

            if not val_done:
                writer.write(tokens, "val")
                val_tokens += len(tokens)
                if val_tokens >= val_target:
                    writer.flush("val")
                    manifest["val_files"] += writer.files
                    writer.files = []
                    val_done = True
                continue

            writer.write(tokens, "train")
            train_tokens += len(tokens)

            now = time.time()
            if now - last_report > 30:
                rate = total_bytes / 1e6 / (now - start)
                print(f"  {train_tokens / 1e9:.2f}B / {train_target / 1e9:.2f}B training tokens, "
                      f"{docs_consumed:,} docs, {rate:.1f} MB/s")
                last_report = now

            if train_tokens >= train_target:
                break

        # Stop the workers before the generator is abandoned.
        pool.terminate()

    writer.flush("train" if val_done else "val")
    manifest["train_files" if val_done else "val_files"] += writer.files
    manifest.update(docs_consumed=docs_consumed, train_tokens=train_tokens,
                    val_tokens=val_tokens, next_shard=writer.index)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed / 60:.1f} min: {train_tokens:,} training tokens, {val_tokens:,} validation tokens, "
          f"{docs_consumed:,} documents")
    if total_bytes:
        print(f"Processed {total_bytes / 1e9:.2f} GB at {total_bytes / 1e6 / elapsed:.1f} MB/s")

    # Sanity check: decode the start of the validation shard.
    first = np.load(os.path.join(args.out_dir, os.path.basename(manifest["val_files"][0])), mmap_mode="r")
    print("\nStart of validation shard:")
    print(repr(tokenizer.decode(first[:80].tolist(), skip_special_tokens=False)))
    assert first[0] == tokenizer.bos_id, "Shards should start with [BOS]"
    assert int(first.max()) < tokenizer.vocab_size


if __name__ == "__main__":
    main()
