"""
Evaluate the trained model against GPT-2 (small, medium, large) on identical data.

    pip install transformers
    python evaluate.py                                    # everything, all models
    python evaluate.py --models ours gpt2 --tasks bpb     # a subset

Tasks
  bpb        Bits per byte, loss per token and perplexity on the same raw held-out
             FineWeb-Edu text. Each model uses its own tokenizer, so only bits per
             byte is directly comparable across models.
  hellaswag  HellaSwag validation accuracy (10,042 examples): pick the most likely of
             four endings. "acc" sums the ending's loss, "acc_norm" averages it per token.
  generate   The same prompts for every model, with greedy decoding, temperature
             sampling and top-p sampling.
  speed      Generation speed in tokens/s, with and without the KV cache.

Fairness
  - Held-out text comes from after the documents used for training (and for the
    tokenizer), so no model has seen it.
  - Every document is scored the same way for every model: it starts with that
    model's start token ([BOS] for ours, <|endoftext|> for GPT-2), and long
    documents are split into non-overlapping windows of the context length.
  - For HellaSwag, GPT-2 is scored without a start token, the setup behind its
    published numbers (124M: ~29.5% acc_norm), so the harness can be checked
    against them. Our model gets [BOS], as every training document started with it.

Results go to results/eval_results.json and results/eval_report.md.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Tokenizer"))

from byte_level_bpe import ByteLevelBPETokenizer  # noqa: E402
from Decoder_Blocks.transformer_model import TransformerModel  # noqa: E402

GPT2_REPOS = {"gpt2": "openai-community/gpt2", "gpt2-medium": "openai-community/gpt2-medium",
              "gpt2-large": "openai-community/gpt2-large"}
# HellaSwag validation set, from the Parquet copy Hugging Face keeps of Rowan/hellaswag.
HELLASWAG_REPO = "Rowan/hellaswag"
HELLASWAG_FILE = "default/validation/0000.parquet"
HELLASWAG_REVISION = "refs/convert/parquet"

PROMPTS = {
    "factual": ["The capital of France is", "Water is made of two elements:"],
    "explanatory": ["Photosynthesis is the process by which", "The seasons change because"],
    "creative": ["Once upon a time, in a small village by the sea,"],
    "completion": ["The most important thing about learning is"],
}
DECODING = {
    "greedy": {"temperature": 0.0},
    "temperature 0.8, top-k 50": {"temperature": 0.8, "top_k": 50},
    "temperature 0.8, top-p 0.9": {"temperature": 0.8, "top_p": 0.9},
}


# ============================================================
# Model wrappers: one interface for our model and GPT-2
# ============================================================

class OurModel:
    def __init__(self, checkpoint, tokenizer_path, device):
        self.tok = ByteLevelBPETokenizer.load(tokenizer_path)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        c = ckpt["config"]
        self.model = TransformerModel(vocab_size=self.tok.vocab_size, d_model=c["d_model"],
                                      padding_id=self.tok.pad_id, num_heads=c["n_heads"], d_ff=c["d_ff"],
                                      max_seq_len=c["seq_len"], num_layers=c["n_layers"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device).eval()
        self.name = f"Ours ({sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M)"
        self.max_len = c["seq_len"]
        self.start_id = self.tok.bos_id
        self.stop_id = self.tok.eos_id
        self.params = sum(p.numel() for p in self.model.parameters())

    def encode(self, text):
        return self.tok.encode(text)

    def decode(self, ids):
        return self.tok.decode(ids)

    def logits(self, ids):
        return self.model(ids)

    def step(self, ids, cache):
        return self.model(ids, past_kv=cache, use_cache=True)


class GPT2Model:
    def __init__(self, key, device, model=None, tokenizer=None):
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(GPT2_REPOS[key])
            model = AutoModelForCausalLM.from_pretrained(GPT2_REPOS[key])
        self.tok = tokenizer
        self.tok.model_max_length = 10**9  # we split long documents ourselves
        self.model = model.to(device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        self.name = f"GPT-2 {key.replace('gpt2', '').strip('-') or 'small'} ({self.params / 1e6:.0f}M)"
        self.max_len = self.model.config.n_positions
        self.start_id = self.model.config.eos_token_id  # <|endoftext|> also separates documents in GPT-2
        self.stop_id = self.model.config.eos_token_id

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids):
        return self.tok.decode(ids)

    def logits(self, ids):
        return self.model(input_ids=ids).logits

    def step(self, ids, cache):
        out = self.model(input_ids=ids, past_key_values=cache, use_cache=True)
        return out.logits, out.past_key_values


# ============================================================
# Data
# ============================================================

def heldout_documents(args):
    cache = os.path.join(args.cache_dir, f"fineweb_heldout_skip{args.skip_docs}_{args.text_mb}mb.jsonl")
    if os.path.exists(cache):
        return [json.loads(line)["text"] for line in open(cache, encoding="utf-8")]

    manifest = os.path.join(args.data_dir, "manifest.json")
    if os.path.exists(manifest):
        used = json.load(open(manifest))["docs_consumed"]
        if args.skip_docs <= used:
            raise ValueError(f"--skip-docs must exceed the {used:,} documents used for training")

    from datasets import load_dataset

    print(f"  streaming FineWeb-Edu, skipping {args.skip_docs:,} documents (a few minutes, cached afterwards)...")
    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    docs, total = [], 0
    for row in stream.skip(args.skip_docs):
        docs.append(row["text"])
        total += len(row["text"].encode("utf-8"))
        if total >= args.text_mb * 1e6:
            break
    os.makedirs(args.cache_dir, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d}) + "\n")
    return docs


def hellaswag_examples(args):
    path = args.hellaswag_file or os.path.join(args.cache_dir, "hellaswag_val.jsonl")
    if not os.path.exists(path):
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        print("  downloading HellaSwag validation set...")
        parquet = hf_hub_download(HELLASWAG_REPO, HELLASWAG_FILE, repo_type="dataset", revision=HELLASWAG_REVISION)
        rows = pq.read_table(parquet).to_pylist()
        os.makedirs(args.cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({"ctx": row["ctx"], "endings": list(row["endings"]),
                                    "label": int(row["label"])}) + "\n")
    examples = [json.loads(line) for line in open(path, encoding="utf-8")]
    if not args.hellaswag_file and len(examples) != 10042:
        raise ValueError(f"Expected 10,042 HellaSwag validation examples, got {len(examples):,}")
    return examples[:args.hellaswag_limit] if args.hellaswag_limit else examples


# ============================================================
# Tasks
# ============================================================

def masked_loss_sums(m, sequences, targets_from, device, autocast):
    """Right-pad a batch, run the model once, and return per-sequence (loss sum, token count)
    over target positions >= targets_from[i]. Right padding is safe with causal attention."""
    width = max(len(s) for s in sequences) - 1
    x = torch.zeros(len(sequences), width, dtype=torch.long)
    y = torch.full((len(sequences), width), -100, dtype=torch.long)
    for i, (s, start) in enumerate(zip(sequences, targets_from)):
        s = torch.tensor(s)
        n = len(s) - 1
        x[i, :n] = s[:-1]
        y[i, :n] = s[1:]
        y[i, :start] = -100  # only score the requested positions
    x, y = x.to(device), y.to(device)
    with torch.no_grad(), autocast:
        logits = m.logits(x)
    losses = F.cross_entropy(logits.float().transpose(1, 2), y, ignore_index=-100, reduction="none")
    mask = (y != -100)
    return (losses * mask).sum(1).double().cpu(), mask.sum(1).cpu()


def eval_bpb(m, docs, device, autocast, batch_size):
    windows, n_bytes = [], 0
    for doc in docs:
        n_bytes += len(doc.encode("utf-8"))
        s = [m.start_id] + m.encode(doc)  # every document token is a target exactly once
        for i in range(0, len(s) - 1, m.max_len):
            windows.append(s[i:i + m.max_len + 1])
    windows.sort(key=len)

    total_loss, total_tokens = 0.0, 0
    start = time.time()
    for b in range(0, len(windows), batch_size):
        batch = windows[b:b + batch_size]
        loss, count = masked_loss_sums(m, batch, [0] * len(batch), device, autocast)
        total_loss += loss.sum().item()
        total_tokens += int(count.sum())
    loss = total_loss / total_tokens
    return {"tokens": total_tokens, "bytes": n_bytes, "bytes_per_token": n_bytes / total_tokens,
            "loss_per_token": loss, "perplexity": math.exp(loss),
            "bits_per_byte": total_loss / math.log(2) / n_bytes, "seconds": time.time() - start}


def eval_hellaswag(m, examples, device, autocast, use_start):
    correct = correct_norm = 0
    for ex in examples:
        prefix = ([m.start_id] if use_start else []) + m.encode(ex["ctx"])
        seqs = [prefix + m.encode(" " + ending) for ending in ex["endings"]]
        # Score only the ending tokens: targets from the last context position onward.
        loss_sum, count = masked_loss_sums(m, seqs, [len(prefix) - 1] * 4, device, autocast)
        correct += int(loss_sum.argmin()) == int(ex["label"])
        correct_norm += int((loss_sum / count).argmin()) == int(ex["label"])
    n = len(examples)
    return {"examples": n, "acc": correct / n, "acc_norm": correct_norm / n}


def sample_next(logits, temperature=0.0, top_k=None, top_p=None, generator=None):
    logits = logits.float()
    if temperature == 0.0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k:
        values, indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float("-inf")).scatter(0, indices, values)
    if top_p:
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, -1)
        remove = probs.cumsum(-1) - probs > top_p  # keep the smallest set reaching top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, order, sorted_logits)
    probs = torch.softmax(logits, -1)
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate(m, prompt, max_new, device, autocast, use_cache=True, seed=0, **decoding):
    ids = [m.start_id] + m.encode(prompt)
    prompt_len = len(ids)
    gen = torch.Generator(device=device).manual_seed(seed)
    cache = None
    with autocast:
        for _ in range(max_new):
            if len(ids) >= m.max_len:
                break
            if use_cache:
                new = ids if cache is None else ids[-1:]
                logits, cache = m.step(torch.tensor([new], device=device), cache)
            else:
                logits = m.logits(torch.tensor([ids], device=device))  # recompute everything
            next_id = sample_next(logits[0, -1], generator=gen, **decoding)
            ids.append(next_id)
            if next_id == m.stop_id:
                break
    return m.decode(ids[prompt_len:]), len(ids) - prompt_len


def eval_speed(m, device, autocast, new_tokens):
    out = {}
    for use_cache in (True, False):
        generate(m, "The", 8, device, autocast, use_cache=use_cache)  # warm-up
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        _, n = generate(m, "The history of science shows that", new_tokens, device, autocast, use_cache=use_cache)
        if device.type == "cuda":
            torch.cuda.synchronize()
        out["with_cache" if use_cache else "without_cache"] = n / (time.time() - start)
    out["speedup"] = out["with_cache"] / out["without_cache"]
    return out


# ============================================================
# Report
# ============================================================

def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def write_report(results, path):
    parts = ["# Evaluation: our model vs GPT-2"]
    names = list(results["models"])

    if any("bpb" in results["models"][n] for n in names):
        rows = []
        for n in names:
            r = results["models"][n].get("bpb")
            if r:
                rows.append([n, f"{r['bits_per_byte']:.4f}", f"{r['loss_per_token']:.4f}", f"{r['perplexity']:.2f}",
                             f"{r['bytes_per_token']:.2f}", f"{r['tokens']:,}"])
        parts += ["## Held-out FineWeb-Edu text", results.get("bpb_note", ""),
                  table(["Model", "Bits/byte (lower is better)", "Loss/token", "Perplexity", "Bytes/token", "Tokens"], rows),
                  "Loss per token and perplexity depend on each model's tokenizer; compare models by bits per byte."]

    if any("hellaswag" in results["models"][n] for n in names):
        rows = [[n, f"{r['acc']:.2%}", f"{r['acc_norm']:.2%}", r["examples"]]
                for n in names if (r := results["models"][n].get("hellaswag"))]
        parts += ["## HellaSwag (validation)", table(["Model", "acc", "acc_norm", "Examples"], rows),
                  "Random guessing scores 25%."]

    if any("speed" in results["models"][n] for n in names):
        rows = [[n, f"{r['with_cache']:.1f}", f"{r['without_cache']:.1f}", f"{r['speedup']:.1f}x"]
                for n in names if (r := results["models"][n].get("speed"))]
        parts += ["## Generation speed (batch 1, tokens/s)",
                  table(["Model", "With KV cache", "Without cache", "Speedup"], rows)]

    if any("generate" in results["models"][n] for n in names):
        parts.append("## Generations")
        first = next(results["models"][n]["generate"] for n in names if "generate" in results["models"][n])
        for category, prompts in first.items():
            for prompt in prompts:
                parts.append(f"### {category}: \"{prompt}\"")
                for method in DECODING:
                    parts.append(f"**{method}**")
                    for n in names:
                        g = results["models"][n].get("generate")
                        if g:
                            parts.append(f"- *{n}:* {g[category][prompt][method]!r}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(p for p in parts if p) + "\n")


# ============================================================
# Main
# ============================================================

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=["ours", "gpt2", "gpt2-medium", "gpt2-large"])
    p.add_argument("--tasks", nargs="*", default=["bpb", "hellaswag", "generate", "speed"])
    p.add_argument("--checkpoint", default="artifacts/final_model.pt")
    p.add_argument("--tokenizer", default="Tokenizer/tokenizer.json")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--cache-dir", default="eval_data")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--skip-docs", type=int, default=2_500_000)
    p.add_argument("--text-mb", type=float, default=5)
    p.add_argument("--text-file", help="Use this .jsonl ({'text': ...} per line) as held-out text instead.")
    p.add_argument("--hellaswag-file", help="Local hellaswag_val.jsonl instead of downloading it.")
    p.add_argument("--hellaswag-limit", type=int, default=None, help="Evaluate only the first N examples.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--speed-tokens", type=int, default=200)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default=None)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = args.dtype or ("bfloat16" if device.type == "cuda" else "float32")
    autocast = torch.autocast(device.type, dtype=torch.bfloat16) if dtype == "bfloat16" else nullcontext()
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "eval_results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {"models": {}}

    print(f"Device: {device} | dtype: {dtype}")
    docs = examples = None
    if "bpb" in args.tasks:
        print("Loading held-out text...")
        docs = ([json.loads(l)["text"] for l in open(args.text_file, encoding="utf-8")]
                if args.text_file else heldout_documents(args))
        size = sum(len(d.encode("utf-8")) for d in docs) / 1e6
        print(f"  {len(docs):,} documents, {size:.1f} MB")
        results["bpb_note"] = (f"{len(docs):,} documents ({size:.1f} MB) from `{args.text_file}`." if args.text_file else
                               f"{len(docs):,} documents ({size:.1f} MB) of FineWeb-Edu `sample-10BT`, "
                               f"starting after document {args.skip_docs:,} (training used the earlier documents).")
    if "hellaswag" in args.tasks:
        examples = hellaswag_examples(args)

    for key in args.models:
        print(f"\n=== {key} ===")
        m = OurModel(args.checkpoint, args.tokenizer, device) if key == "ours" else GPT2Model(key, device)
        r = results["models"].setdefault(m.name, {"parameters": m.params})

        if "bpb" in args.tasks:
            r["bpb"] = eval_bpb(m, docs, device, autocast, args.batch_size)
            print(f"  bits/byte {r['bpb']['bits_per_byte']:.4f} | loss/token {r['bpb']['loss_per_token']:.4f} | "
                  f"perplexity {r['bpb']['perplexity']:.2f}")
        if "hellaswag" in args.tasks:
            r["hellaswag"] = eval_hellaswag(m, examples, device, autocast, use_start=(key == "ours"))
            print(f"  HellaSwag acc {r['hellaswag']['acc']:.2%} | acc_norm {r['hellaswag']['acc_norm']:.2%} "
                  f"({r['hellaswag']['examples']:,} examples)")
        if "speed" in args.tasks:
            r["speed"] = eval_speed(m, device, autocast, args.speed_tokens)
            print(f"  speed {r['speed']['with_cache']:.1f} tok/s with cache, "
                  f"{r['speed']['without_cache']:.1f} without ({r['speed']['speedup']:.1f}x)")
        if "generate" in args.tasks:
            r["generate"] = {cat: {pr: {name: generate(m, pr, args.max_new_tokens, device, autocast, seed=i, **cfg)[0]
                                        for i, (name, cfg) in enumerate(DECODING.items())}
                                   for pr in prompts} for cat, prompts in PROMPTS.items()}
            print(f"  generated {sum(len(v) for v in PROMPTS.values()) * len(DECODING)} completions")

        with open(results_path, "w") as f:  # save after every model
            json.dump(results, f, indent=2)
        del m
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = os.path.join(args.out_dir, "eval_report.md")
    write_report(results, report)
    print(f"\nWrote {results_path} and {report}")


if __name__ == "__main__":
    main()
