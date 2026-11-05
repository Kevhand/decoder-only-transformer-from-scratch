"""
Check the tokenizer and the tokenized shards before training. From the project root:

    python check_data.py --data-dir data --tokenizer Tokenizer/tokenizer.json

Verifies that every shard listed in the manifest is present and complete, that
the token ids are valid, and that this tokenizer.json is exactly the one that
produced the shards (by decoding documents and re-encoding them).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tokenizer"))
from Tokenizer.byte_level_bpe import ByteLevelBPETokenizer  # noqa: E402


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAIL  {message}")
    print(f"PASS  {message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--tokenizer", default="Tokenizer/tokenizer.json")
    parser.add_argument("--reencode-docs", type=int, default=300)
    args = parser.parse_args()

    tok = ByteLevelBPETokenizer.load(args.tokenizer)
    with open(os.path.join(args.data_dir, "manifest.json")) as f:
        m = json.load(f)

    # ---- tokenizer ----
    print("Tokenizer")
    check(tok.vocab_size == len(tok.vocab) == 260 + len(tok.merges),
          f"vocab size {tok.vocab_size:,} = 256 bytes + 4 special tokens + {len(tok.merges):,} merges")
    check(tok.vocab_size <= 65536, "token ids fit in uint16")
    check((tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id) == (256, 257, 258, 259),
          "special ids: PAD 256, BOS 257, EOS 258, UNK 259")
    sample = "Jane Austen’s novels weren't only about romance. 2026: 12.5% — 北京 🚀"
    check(tok.decode(tok.encode(sample)) == sample, "round trip on mixed text, punctuation, Unicode and emoji")
    check((m["vocab_size"], m["tokenizer_merges"]) == (tok.vocab_size, len(tok.merges)),
          "manifest records the same vocab size and merge count")
    check((m["pad_id"], m["bos_id"], m["eos_id"]) == (tok.pad_id, tok.bos_id, tok.eos_id),
          "manifest records the same special ids")

    # ---- shards ----
    print("\nShards")
    paths = {split: [os.path.join(args.data_dir, os.path.basename(p)) for p in m[f"{split}_files"]]
             for split in ("val", "train")}
    missing = [p for ps in paths.values() for p in ps if not os.path.exists(p)]
    check(not missing, f"all {sum(map(len, paths.values()))} shard files present" +
          (f" (missing: {missing})" if missing else ""))

    for split in ("val", "train"):
        total, max_id, pads, bos, eos = 0, 0, 0, 0, 0
        for p in paths[split]:
            shard = np.load(p, mmap_mode="r")
            if shard.dtype != np.uint16:
                raise SystemExit(f"FAIL  {p} has dtype {shard.dtype}, expected uint16")
            total += len(shard)
            max_id = max(max_id, int(shard.max()))
            counts = np.bincount(shard, minlength=tok.vocab_size)
            pads += int(counts[tok.pad_id])
            bos += int(counts[tok.bos_id])
            eos += int(counts[tok.eos_id])
        expected = m[f"{split}_tokens"]
        check(total == expected, f"{split}: {total:,} tokens on disk = {expected:,} in the manifest")
        check(max_id < tok.vocab_size, f"{split}: largest token id {max_id:,} < vocab size")
        check(pads == 0, f"{split}: no [PAD] tokens (documents are packed, not padded)")
        check(abs(bos - eos) <= 1, f"{split}: {bos:,} documents ([BOS] {bos:,}, [EOS] {eos:,})")

    # ---- same tokenizer as the shards ----
    print("\nTokenizer matches the shards")
    val = np.load(paths["val"][0], mmap_mode="r")
    check(val[0] == tok.bos_id, "validation shard starts with [BOS]")

    starts = np.flatnonzero(val[:5_000_000] == tok.bos_id)[:args.reencode_docs + 1]
    checked = mismatched = 0
    for a, b in zip(starts[:-1], starts[1:]):
        doc = val[a:b].tolist()
        if doc[-1] != tok.eos_id:
            raise SystemExit("FAIL  a document does not end with [EOS]")
        content = doc[1:-1]
        if tok.encode(tok.decode(content)) != content:
            mismatched += 1
        checked += 1
    check(mismatched == 0, f"{checked} documents decoded and re-encoded to identical ids "
                           f"(this tokenizer.json produced these shards)")

    print("\nFirst document:")
    print(repr(tok.decode(val[starts[0]:starts[1]].tolist())[:300]))
    print("\nAll data checks passed.")


if __name__ == "__main__":
    main()