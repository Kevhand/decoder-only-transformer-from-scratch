"""
Compare your byte-level BPE tokenizer against GPT-2, LLaMA-2 (32k) and GPT-4 (cl100k).

    pip install datasets tiktoken transformers sentencepiece
    python compare_tokenizers.py --tokenizer tokenizer.json

Evaluation sets:
  fineweb_heldout  FineWeb-Edu documents from far past the tokenizer's training sample
  code             Python standard-library source (no download needed)
  hindi            Hindi Wikipedia articles
  numbers          Generated number-heavy text (prices, dates, decimals)

Metrics:
  bytes/token   UTF-8 bytes divided by tokens; higher means better compression
  round-trip    share of documents where decode(encode(text)) == text exactly
  vocab used    share of the vocabulary that appears when encoding the FineWeb set
  encode MB/s   single-process encoding speed on the FineWeb set

Results are printed and also written to tokenizer_comparison.md and .json.
A tokenizer whose package is missing is skipped with a message.
"""

import argparse
import importlib
import inspect
import json
import random
import time

from byte_level_bpe import ByteLevelBPETokenizer


# ============================================================
# Evaluation sets
# ============================================================

def take_until(texts, max_bytes):
    docs, total = [], 0
    for text in texts:
        if not text.strip():
            continue
        docs.append(text)
        total += len(text.encode("utf-8"))
        if total >= max_bytes:
            break
    return docs


def fineweb_heldout(skip_docs, max_bytes):
    from datasets import load_dataset

    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    print(f"  skipping the first {skip_docs:,} FineWeb-Edu documents (tokenizer training region)...")
    return take_until((row["text"] for row in stream.skip(skip_docs)), max_bytes)


def hindi_wikipedia(max_bytes):
    from datasets import load_dataset

    stream = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)
    return take_until((row["text"] for row in stream), max_bytes)


def python_code(max_bytes):
    modules = [
        "argparse", "collections", "json.decoder", "json.encoder", "logging", "pathlib",
        "subprocess", "threading", "typing", "unittest.case", "email.message", "http.client",
        "tarfile", "zipfile", "csv", "functools", "dataclasses", "textwrap", "string", "heapq",
    ]
    sources = []
    for name in modules:
        try:
            sources.append(inspect.getsource(importlib.import_module(name)))
        except (OSError, TypeError, ImportError):
            pass
    return take_until(sources, max_bytes)


def numbers_text(max_bytes, seed=0):
    rng = random.Random(seed)
    templates = [
        lambda: f"In {rng.randint(1900, 2030)}, revenue rose {rng.uniform(0, 40):.1f}% to ${rng.randint(1_000, 9_999_999):,}.",
        lambda: f"The sample measured {rng.uniform(0, 100):.3f} mm at {rng.randint(-40, 120)} degrees on "
                f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1990, 2030)}.",
        lambda: f"Order #{rng.randint(10_000, 999_999)} contains {rng.randint(1, 500)} units at "
                f"{rng.uniform(0, 999):.2f} each, total {rng.uniform(0, 99_999):.2f}.",
        lambda: f"{rng.randint(0, 999)} + {rng.randint(0, 999)} = {rng.randint(0, 1999)}; "
                f"{rng.randint(2, 99)} x {rng.randint(2, 99)} = {rng.randint(4, 9801)}.",
        lambda: f"Call +91 {rng.randint(70000, 99999)} {rng.randint(10000, 99999)} before {rng.randint(1, 12)}:"
                f"{rng.randint(0, 59):02d} on {rng.randint(1, 31)} March {rng.randint(2000, 2030)}.",
    ]
    docs = []
    while sum(len(d) for d in docs) < max_bytes:
        docs.append(" ".join(rng.choice(templates)() for _ in range(20)))
    return docs


# ============================================================
# Tokenizer wrappers: name, encode(text) -> ids, decode(ids) -> text,
# piece(id) -> readable token, vocab_size
# ============================================================

def show_bytes(raw):
    """Readable form of one token's bytes; partial UTF-8 characters are shown as hex."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return "<" + " ".join(f"{b:02x}" for b in raw) + ">"


class Mine:
    def __init__(self, path):
        self.tok = ByteLevelBPETokenizer.load(path)
        self.name = f"Ours ({self.tok.vocab_size // 1000}k)"
        self.vocab_size = self.tok.vocab_size

    def encode(self, text):
        return self.tok.encode(text)

    def decode(self, ids):
        return self.tok.decode(ids)

    def piece(self, token_id):
        return show_bytes(self.tok.itos[token_id])


class Tiktoken:
    def __init__(self, encoding, label):
        import tiktoken

        self.enc = tiktoken.get_encoding(encoding)
        self.name = label
        self.vocab_size = self.enc.n_vocab

    def encode(self, text):
        # encode_ordinary treats "<|endoftext|>" in the data as plain text.
        return self.enc.encode_ordinary(text)

    def decode(self, ids):
        return self.enc.decode(ids)

    def piece(self, token_id):
        return show_bytes(self.enc.decode_single_token_bytes(token_id))


class HFTokenizer:
    def __init__(self, repo, label):
        from transformers import AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # silence "sequence longer than max length" warnings
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.name = label
        self.vocab_size = len(self.tok)

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def piece(self, token_id):
        # SentencePiece shows a leading space as "▁" and byte fallback as "<0xE0>".
        return self.tok.convert_ids_to_tokens(token_id)


def load_tokenizers(args):
    candidates = [
        ("ours", lambda: Mine(args.tokenizer)),
        ("gpt2", lambda: Tiktoken("gpt2", "GPT-2 (50k)")),
        ("llama2", lambda: HFTokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "LLaMA-2 (32k)")),
        ("cl100k", lambda: Tiktoken("cl100k_base", "GPT-4 cl100k (100k)")),
    ]
    tokenizers = []
    for key, make in candidates:
        if key not in args.only:
            continue
        try:
            tokenizers.append(make())
        except Exception as e:  # missing package, no network, etc.
            print(f"  skipping {key}: {type(e).__name__}: {e}")
    return tokenizers


# ============================================================
# Evaluation
# ============================================================

def evaluate(tokenizer, docs, measure_speed=False):
    n_bytes = sum(len(d.encode("utf-8")) for d in docs)
    n_tokens = 0
    exact = 0
    used_ids = set()

    start = time.perf_counter()
    encoded = [tokenizer.encode(d) for d in docs]
    seconds = time.perf_counter() - start

    for doc, ids in zip(docs, encoded):
        n_tokens += len(ids)
        used_ids.update(ids)
        if tokenizer.decode(ids) == doc:
            exact += 1

    result = {
        "bytes_per_token": n_bytes / max(n_tokens, 1),
        "tokens": n_tokens,
        "round_trip": exact / len(docs),
    }
    if measure_speed:
        result["vocab_used"] = len(used_ids) / tokenizer.vocab_size
        result["encode_mb_s"] = n_bytes / 1e6 / seconds
    return result


def show_examples(tokenizers, sentences):
    lines = []
    for sentence in sentences:
        lines.append(f"\n{sentence!r}")
        for tok in tokenizers:
            pieces = [tok.piece(i) for i in tok.encode(sentence)]
            lines.append(f"  {tok.name:<22} {len(pieces):>3} tokens  {pieces}")
    return "\n".join(lines)


def markdown_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--skip-docs", type=int, default=70_000,
                        help="FineWeb-Edu docs to skip; must exceed the docs used to train the tokenizer.")
    parser.add_argument("--fineweb-mb", type=float, default=5)
    parser.add_argument("--other-mb", type=float, default=1)
    parser.add_argument("--heldout-file", help="Use a local text file instead of streaming FineWeb-Edu.")
    parser.add_argument("--only", nargs="*", default=["ours", "gpt2", "llama2", "cl100k"])
    parser.add_argument("--sets", nargs="*", default=["fineweb_heldout", "code", "hindi", "numbers"])
    args = parser.parse_args()

    print("Loading tokenizers...")
    tokenizers = load_tokenizers(args)

    print("Loading evaluation sets...")
    loaders = {
        "fineweb_heldout": (lambda: take_until(open(args.heldout_file, encoding="utf-8").read().split("\n\n"),
                                               args.fineweb_mb * 1e6)
                            if args.heldout_file else fineweb_heldout(args.skip_docs, args.fineweb_mb * 1e6)),
        "code": lambda: python_code(args.other_mb * 1e6),
        "hindi": lambda: hindi_wikipedia(args.other_mb * 1e6),
        "numbers": lambda: numbers_text(args.other_mb * 1e6),
    }
    sets = {}
    for name in args.sets:
        try:
            sets[name] = loaders[name]()
            size = sum(len(d.encode("utf-8")) for d in sets[name]) / 1e6
            print(f"  {name}: {len(sets[name]):,} docs, {size:.1f} MB")
        except Exception as e:
            print(f"  skipping set {name}: {type(e).__name__}: {e}")

    results = {}
    for tok in tokenizers:
        results[tok.name] = {}
        for set_name, docs in sets.items():
            print(f"Evaluating {tok.name} on {set_name}...")
            results[tok.name][set_name] = evaluate(tok, docs, measure_speed=(set_name == "fineweb_heldout"))

    # Table 1: compression on every set
    headers = ["Tokenizer", "Vocab"] + [f"{s} bytes/token" for s in sets]
    rows = [[tok.name, f"{tok.vocab_size:,}"] +
            [f"{results[tok.name][s]['bytes_per_token']:.2f}" for s in sets] for tok in tokenizers]
    compression = markdown_table(headers, rows)

    # Table 2: round-trip exactness on every set
    headers = ["Tokenizer"] + [f"{s} round-trip" for s in sets]
    rows = [[tok.name] + [f"{results[tok.name][s]['round_trip']:.1%}" for s in sets] for tok in tokenizers]
    round_trip = markdown_table(headers, rows)

    # Table 3: vocab use and speed on FineWeb
    extra = ""
    if "fineweb_heldout" in sets:
        headers = ["Tokenizer", "Vocab used (FineWeb)", "Encode MB/s (1 process)"]
        rows = [[tok.name,
                 f"{results[tok.name]['fineweb_heldout']['vocab_used']:.1%}",
                 f"{results[tok.name]['fineweb_heldout']['encode_mb_s']:.2f}"] for tok in tokenizers]
        extra = markdown_table(headers, rows)

    examples = show_examples(tokenizers, [
        "Jane Austen’s novels weren't only about romance; they're about independence.",
        "In 2026, revenue rose 12.5% to $3,450,000 across 1,024 stores.",
        "मशीन लर्निंग कंप्यूटर को डेटा से सीखना सिखाती है।",
        "def train(model, data):\n    for batch in data:\n        loss = model(batch)",
    ])

    report = "\n\n".join([
        "## Compression (higher bytes/token is better)", compression,
        "## Round-trip exactness", round_trip,
        "## Vocabulary use and speed", extra,
        "## Example splits", "```" + examples + "\n```",
    ])
    print("\n" + report)

    with open("tokenizer_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Tokenizer comparison\n\n" + report + "\n")
    with open("tokenizer_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved tokenizer_comparison.md and tokenizer_comparison.json")


if __name__ == "__main__":
    main()
