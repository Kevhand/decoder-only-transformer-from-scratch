"""
Check the KV cache is exact, and measure generation speed with and without it at
increasing context lengths (the cache's advantage grows with length).

    python check_kv_cache.py --label before
    python check_kv_cache.py --label after

1. Correctness (float32): greedy generation with the cache must produce the same tokens
   as recomputing the whole sequence every step, with logits matching to 1e-3.
   Exits with an error if not.
2. Speed (bfloat16, batch 1): generate until the context is full (1,024 tokens), timing
   each step, and report tokens/s for each quarter of the context.

Writes results/kv_cache_<label>.json, and results/kv_cache_report.md comparing
"before" and "after" when both exist.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import OurModel  # noqa: E402

PROMPT = "The history of science shows that"


def finite_diff(a, b):
    # The PAD logit is -inf in both; compare the rest.
    ok = torch.isfinite(a) & torch.isfinite(b)
    return (a[ok] - b[ok]).abs().max().item()


@torch.no_grad()
def correctness(m, device, steps):
    ids = [m.start_id] + m.encode(PROMPT)
    logits, cache = m.step(torch.tensor([ids], device=device), None)
    worst, same = 0.0, True
    for _ in range(steps):
        full = m.logits(torch.tensor([ids], device=device))[0, -1].float()
        cached = logits[0, -1].float()
        worst = max(worst, finite_diff(cached, full))
        nxt_cached, nxt_full = int(cached.argmax()), int(full.argmax())
        same &= nxt_cached == nxt_full
        ids.append(nxt_full)
        logits, cache = m.step(torch.tensor([[nxt_full]], device=device), cache)
    return {"steps": steps, "max_logit_diff": worst, "same_tokens": same}


@torch.no_grad()
def speed(m, device, use_cache, autocast):
    ids = [m.start_id] + m.encode(PROMPT)
    start_len = len(ids)
    stamps = []
    cache = None
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with autocast:
        while len(ids) < m.max_len:
            if use_cache:
                new = ids if cache is None else ids[-1:]
                logits, cache = m.step(torch.tensor([new], device=device), cache)
            else:
                logits = m.logits(torch.tensor([ids], device=device))
            ids.append(int(logits[0, -1].float().argmax()))  # .item() waits for the GPU
            stamps.append((len(ids), time.time() - t0))
    # tokens/s within each quarter of the context
    quarters = []
    edges = [start_len] + [m.max_len * q // 4 for q in (1, 2, 3, 4)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = [t for n, t in stamps if lo < n <= hi]
        before = [t for n, t in stamps if n <= lo]
        if len(inside) > 1:
            span = inside[-1] - (before[-1] if before else 0.0)
            quarters.append({"context": f"{max(lo, 0)}-{hi}", "tokens_per_s": len(inside) / span})
    total = stamps[-1][1]
    return {"tokens": len(stamps), "seconds": total, "tokens_per_s": len(stamps) / total, "quarters": quarters}


def write_report(out_dir):
    runs = {}
    for label in ("before", "after"):
        path = os.path.join(out_dir, f"kv_cache_{label}.json")
        if os.path.exists(path):
            runs[label] = json.load(open(path))
    lines = ["# KV cache: correctness and generation speed", "",
             "Batch 1, bfloat16, greedy, prompt then generation until the 1,024-token context is full. "
             "Tokens/s within each part of the context. \"Before\"/\"after\" = without/with the fix that skips "
             "the attention mask when one new token is fed (the no-cache path doesn't change, so its numbers "
             "show the run-to-run noise).", ""]
    names = sorted({n for r in runs.values() for n in r["models"]})
    for n in names:
        lines += [f"## {n}", ""]
        for label, r in runs.items():
            c = r["models"].get(n, {}).get("correctness")
            if c:
                lines.append(f"- {label}: cached vs full recompute over {c['steps']} steps (float32): "
                             f"same tokens {'yes' if c['same_tokens'] else 'NO'}, "
                             f"max logit difference {c['max_logit_diff']:.1e}")
        header = ["Context (tokens)"]
        cols = []
        for label in runs:
            for mode in ("cache", "no_cache"):
                if n in runs[label]["models"]:
                    header.append(f"{label}: {'with' if mode == 'cache' else 'without'} cache")
                    cols.append(runs[label]["models"][n]["speed"][mode])
        if "after" in runs and n in runs["after"]["models"]:
            header.append("Speedup from cache (after)")
        lines += ["", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        rows = max(len(c["quarters"]) for c in cols)
        for i in range(rows):
            cells = [cols[0]["quarters"][i]["context"]]
            cells += [f"{c['quarters'][i]['tokens_per_s']:.1f}" for c in cols]
            if "after" in runs and n in runs["after"]["models"]:
                s = runs["after"]["models"][n]["speed"]
                cells.append(f"{s['cache']['quarters'][i]['tokens_per_s'] / s['no_cache']['quarters'][i]['tokens_per_s']:.2f}x")
            lines.append("| " + " | ".join(cells) + " |")
        cells = ["whole run"] + [f"{c['tokens_per_s']:.1f}" for c in cols]
        if "after" in runs and n in runs["after"]["models"]:
            s = runs["after"]["models"][n]["speed"]
            cells.append(f"{s['cache']['tokens_per_s'] / s['no_cache']['tokens_per_s']:.2f}x")
        lines += ["| " + " | ".join(cells) + " |", ""]
    with open(os.path.join(out_dir, "kv_cache_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--checkpoints", nargs="*", default=["artifacts/final_model.pt", "artifacts_350m/final_model.pt"])
    p.add_argument("--tokenizer", default="Tokenizer/tokenizer.json")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--check-steps", type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = torch.autocast(device.type, dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    os.makedirs(args.out_dir, exist_ok=True)
    out = {"label": args.label, "models": {}}
    failed = False

    for ckpt in args.checkpoints:
        m = OurModel(ckpt, args.tokenizer, device)
        print(f"=== {m.name} ===")
        # Exact float32 for the correctness check
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        c = correctness(m, device, args.check_steps)
        ok = c["same_tokens"] and c["max_logit_diff"] < 1e-3
        failed |= not ok
        print(f"  correctness: same tokens {c['same_tokens']}, max logit diff {c['max_logit_diff']:.2e} "
              f"-> {'PASS' if ok else 'FAIL'}")
        speed(m, device, True, bf16)  # warm-up
        s = {"cache": speed(m, device, True, bf16), "no_cache": speed(m, device, False, bf16)}
        for mode, r in s.items():
            print(f"  {mode:<8} " + " | ".join(f"{q['context']}: {q['tokens_per_s']:.1f} tok/s" for q in r["quarters"]))
        out["models"][m.name] = {"correctness": c, "speed": s}
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, f"kv_cache_{args.label}.json"), "w") as f:
        json.dump(out, f, indent=2)
    write_report(args.out_dir)
    print(f"Wrote {args.out_dir}/kv_cache_{args.label}.json and {args.out_dir}/kv_cache_report.md")
    if failed:
        sys.exit("KV cache correctness check FAILED")


if __name__ == "__main__":
    main()
