"""
Bits per byte on several kinds of text, to see where each model is strong or weak.

    python domain_eval.py                       # both of our models + GPT-2 small/medium/large, all domains
    python domain_eval.py --domains books wiki  # a subset

Domains (each is text no model trained on, as far as we can tell):
  fineweb_edu  held-out FineWeb-Edu (after document 7.5M): educational web pages, our training domain
  wiki         WikiText-103 test set: 60 Wikipedia "Good" and "Featured" articles
  books        PG-19 test set: novels and other books from Project Gutenberg (published before 1919)
  web          C4 validation set: general English web pages

Every model reads the same raw text, scored exactly as in evaluate.py (start token,
non-overlapping 1024-token windows). Bits per byte is comparable across tokenizers.

Results go to results/domain_results.json and results/domain_report.md, merged by model name.
Use --file DOMAIN=path.jsonl ({"text": ...} per line) to supply a domain's text yourself.
"""

import argparse
import gc
import gzip
import json
import os
import re
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import GPT2Model, OurModel, eval_bpb, heldout_documents  # noqa: E402

DOMAINS = {
    "fineweb_edu": "FineWeb-Edu (training domain)",
    "wiki": "Wikipedia (WikiText-103)",
    "books": "Books (PG-19)",
    "web": "General web (C4)",
}


# ============================================================
# Loading each domain's text
# ============================================================

def find_file(repo, pattern, revisions=("main", "refs/convert/parquet")):
    """First file in a dataset repo whose path matches the pattern, trying each revision."""
    from huggingface_hub import list_repo_files

    for revision in revisions:
        try:
            files = list_repo_files(repo, repo_type="dataset", revision=revision)
        except Exception:
            continue
        matches = sorted(f for f in files if re.search(pattern, f))
        if matches:
            return matches[0], revision
    return None, None


def download(repo, pattern, revisions=("main", "refs/convert/parquet")):
    from huggingface_hub import hf_hub_download

    path, revision = find_file(repo, pattern, revisions)
    if path is None:
        return None
    print(f"  downloading {repo}/{path}")
    return hf_hub_download(repo, path, repo_type="dataset", revision=revision)


def read_parquet_texts(path):
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=["text"]).column("text").to_pylist()


def wikitext_detokenize(text):
    """Undo WikiText's tokenization (" @-@ ", spaces before punctuation), as lm-evaluation-harness does,
    so every model reads normal text rather than pre-split tokens."""
    text = text.replace("s '", "s'")
    text = re.sub(r"/' [0-9]/", r"/'[0-9]/", text)
    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    for p in (":", ";", ".", "!", "?", ","):
        text = text.replace(f" {p} ", f"{p} ")
    text = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", text)
    text = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", text)
    text = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", text)
    text = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', text)
    text = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", text)
    text = text.replace("= = = =", "====").replace("= = =", "===").replace("= =", "==")
    text = text.replace(" " + chr(176) + " ", chr(176))
    text = text.replace(" \n", "\n").replace("\n ", "\n")
    text = text.replace(" N ", " 1 ")
    text = text.replace(" 's", "'s")
    return text


def load_wiki():
    path = download("Salesforce/wikitext", r"wikitext-103-raw-v1/test")
    if path is None:
        raise RuntimeError("Couldn't find the WikiText-103 test file on Hugging Face.")
    lines = read_parquet_texts(path)
    # Articles start with a top-level heading line " = Title = " (one '=' on each side).
    articles, current = [], []
    for line in lines:
        if re.fullmatch(r" = [^=].* = \n", line) and current:
            articles.append("".join(current))
            current = []
        current.append(line)
    if current:
        articles.append("".join(current))
    return [wikitext_detokenize(a).strip() for a in articles if a.strip()]


def load_books(text_mb, book_kb):
    # deepmind/pg19 is the original; emozilla/pg19-test is a Parquet copy of the same test split.
    path = None
    for repo, pattern in (("deepmind/pg19", r"test.*\.parquet$"), ("emozilla/pg19-test", r"\.parquet$")):
        path = download(repo, pattern)
        if path:
            break
    if path is None:
        raise RuntimeError("Couldn't find the PG-19 test split on Hugging Face; supply it with --file books=...")
    docs, total = [], 0
    for book in read_parquet_texts(path):
        # A slice from 20% into the book, so it's the story rather than the title pages,
        # cut at paragraph breaks.
        start = book.find("\n\n", len(book) // 5)
        end = book.find("\n\n", start + book_kb * 1000)
        piece = book[start:end if end != -1 else None].strip()
        if piece:
            docs.append(piece)
            total += len(piece.encode("utf-8"))
        if total >= text_mb * 1e6:
            break
    return docs


def load_web(text_mb):
    from huggingface_hub import hf_hub_download

    print("  downloading allenai/c4/en/c4-validation.00000-of-00008.json.gz")
    path = hf_hub_download("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", repo_type="dataset")
    docs, total = [], 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            docs.append(text)
            total += len(text.encode("utf-8"))
            if total >= text_mb * 1e6:
                break
    return docs


def load_domain(name, args):
    if name in args.files:
        return [json.loads(l)["text"] for l in open(args.files[name], encoding="utf-8")]

    cache = os.path.join(args.cache_dir, f"domain_{name}.jsonl")
    if os.path.exists(cache):
        return [json.loads(l)["text"] for l in open(cache, encoding="utf-8")]

    if name == "fineweb_edu":
        docs = heldout_documents(SimpleNamespace(data_dir=args.data_dir, cache_dir=args.cache_dir,
                                                 skip_docs=args.skip_docs, text_mb=args.text_mb))
    elif name == "wiki":
        docs = load_wiki()
    elif name == "books":
        docs = load_books(args.text_mb, args.book_kb)
    elif name == "web":
        docs = load_web(args.text_mb)
    else:
        raise ValueError(f"Unknown domain {name}")

    os.makedirs(args.cache_dir, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d}) + "\n")
    return docs


# ============================================================
# Report
# ============================================================

def write_report(results, path):
    domains = [d for d in DOMAINS if d in results["domains"]]
    names = sorted(results["models"], key=lambda n: results["models"][n]["parameters"])
    lines = ["# Bits per byte by kind of text (lower is better)", "",
             "| Model | " + " | ".join(DOMAINS[d] for d in domains) + " |",
             "|---|" + "---|" * len(domains)]
    for n in names:
        r = results["models"][n]
        cells = [f"{r[d]['bits_per_byte']:.4f}" if d in r else "" for d in domains]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "## Text used", ""]
    for d in domains:
        info = results["domains"][d]
        lines.append(f"- **{DOMAINS[d]}**: {info['documents']:,} documents, {info['mb']:.1f} MB. {info['source']}")
    lines += ["", "Each model uses its own tokenizer, so loss per token is not comparable across models; "
                  "bits per byte is (total loss in bits / bytes of UTF-8 text)."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


SOURCES = {
    "fineweb_edu": "HuggingFaceFW/fineweb-edu sample-10BT, after the documents used for training.",
    "wiki": "Salesforce/wikitext, wikitext-103-raw-v1 test split, detokenized as in lm-evaluation-harness.",
    "books": "PG-19 test split; from each book, a slice starting 20% in.",
    "web": "allenai/c4, English validation shard 0.",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=["ours", "gpt2", "gpt2-medium", "gpt2-large"])
    p.add_argument("--checkpoints", nargs="*", default=["artifacts/final_model.pt", "artifacts_350m/final_model.pt"],
                   help="Our model checkpoints, each evaluated when 'ours' is in --models.")
    p.add_argument("--domains", nargs="*", default=list(DOMAINS))
    p.add_argument("--file", action="append", default=[], help="DOMAIN=path.jsonl to supply a domain's text.")
    p.add_argument("--tokenizer", default="Tokenizer/tokenizer.json")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--cache-dir", default="eval_data")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--skip-docs", type=int, default=7_500_000)
    p.add_argument("--text-mb", type=float, default=5, help="Text per domain (WikiText-103 test is smaller).")
    p.add_argument("--book-kb", type=int, default=100, help="Text taken from each book.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default=None)
    args = p.parse_args()
    args.files = dict(item.split("=", 1) for item in args.file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = args.dtype or ("bfloat16" if device.type == "cuda" else "float32")
    autocast = torch.autocast(device.type, dtype=torch.bfloat16) if dtype == "bfloat16" else nullcontext()
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "domain_results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {"domains": {}, "models": {}}

    print(f"Device: {device} | dtype: {dtype}\nLoading text...")
    texts = {}
    for d in args.domains:
        texts[d] = load_domain(d, args)
        mb = sum(len(t.encode("utf-8")) for t in texts[d]) / 1e6
        results["domains"][d] = {"documents": len(texts[d]), "mb": mb, "source": SOURCES.get(d, "")}
        print(f"  {DOMAINS[d]}: {len(texts[d]):,} documents, {mb:.1f} MB")

    jobs = [("ours", c) for c in args.checkpoints] if "ours" in args.models else []
    jobs += [(m, None) for m in args.models if m != "ours"]
    for key, checkpoint in jobs:
        m = OurModel(checkpoint, args.tokenizer, device) if key == "ours" else GPT2Model(key, device)
        print(f"\n=== {m.name} ===")
        r = results["models"].setdefault(m.name, {"parameters": m.params})
        for d in args.domains:
            r[d] = eval_bpb(m, texts[d], device, autocast, args.batch_size)
            print(f"  {DOMAINS[d]:<32} bits/byte {r[d]['bits_per_byte']:.4f}")
        with open(results_path, "w") as f:  # save after every model, so a crash loses little
            json.dump(results, f, indent=2)
        del m
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_report(results, os.path.join(args.out_dir, "domain_report.md"))
    print(f"\nWrote {results_path} and {os.path.join(args.out_dir, 'domain_report.md')}")


if __name__ == "__main__":
    main()
