"""
Collect, verify and (optionally) upload everything from a finished training run.

    python3 export_artifacts.py                         # build and verify artifacts/ only
    python3 export_artifacts.py --repo Kevhand/decoder-110m   # ...and upload to a private HF repo

What it produces in artifacts/:
    final_model.pt   weights only (~0.44 GB) + model config: what evaluation and generation load
    summary.json     model size, training setup and final metrics, for the write-up

What it uploads (files already in the repo are skipped, so it can be rerun):
    final_model.pt, summary.json
    full/latest.pt                 full checkpoint with optimizer state (to continue training)
    intermediate/model_step_*.pt   weights at steps 1000, 2000, ...
    logs/log.jsonl, logs/train_stdout.txt (if present)
    tokenizer/tokenizer.json, data/manifest.json
"""

import argparse
import glob
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Tokenizer"))

from byte_level_bpe import ByteLevelBPETokenizer  # noqa: E402
from Decoder_Blocks.transformer_model import TransformerModel  # noqa: E402


def build_model(config, tokenizer):
    return TransformerModel(vocab_size=tokenizer.vocab_size, d_model=config["d_model"], padding_id=tokenizer.pad_id,
                            num_heads=config["n_heads"], d_ff=config["d_ff"], max_seq_len=config["seq_len"],
                            num_layers=config["n_layers"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--tokenizer", default="Tokenizer/tokenizer.json")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--stdout-log", default="train_stdout.txt", help="Saved terminal output with the samples.")
    p.add_argument("--repo", help="Private Hugging Face model repo to upload to, e.g. Kevhand/decoder-110m")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer.load(args.tokenizer)

    # ---- final checkpoint -> weights-only model file ----
    latest_path = os.path.join(args.ckpt_dir, "latest.pt")
    ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
    config, step = ckpt["config"], ckpt["step"]

    model = build_model(config, tokenizer)
    model.load_state_dict(ckpt["model"])  # fails loudly if anything is missing or mismatched
    n_params = sum(p.numel() for p in model.parameters())

    final_path = os.path.join(args.out, "final_model.pt")
    torch.save({"model": ckpt["model"], "config": config, "step": step}, final_path)
    print(f"Wrote {final_path}: step {step}, {n_params / 1e6:.1f}M parameters, "
          f"{os.path.getsize(final_path) / 1e9:.2f} GB")

    # ---- verify the exported file reproduces the checkpoint ----
    reloaded = build_model(config, tokenizer)
    reloaded.load_state_dict(torch.load(final_path, map_location="cpu", weights_only=False)["model"])
    ids = torch.tensor([[tokenizer.bos_id] + tokenizer.encode("The most important thing about learning is")])
    model.eval(), reloaded.eval()
    with torch.no_grad():
        a, b = model(ids), reloaded(ids)
    assert torch.equal(a, b), "exported weights do not reproduce the checkpoint"
    assert reloaded.lm_head.weight is reloaded.token_embedding.embedding.weight, "weight tying lost"
    print("Verified: exported model gives identical logits, weights still tied")

    # ---- summary from the training log ----
    log_path = os.path.join(args.ckpt_dir, "log.jsonl")
    records = [json.loads(line) for line in open(log_path)]
    vals = [r for r in records if "val_loss" in r]
    trains = [r for r in records if "train_loss" in r]
    with open(os.path.join(args.data_dir, "manifest.json")) as f:
        manifest = json.load(f)

    summary = {
        "parameters": n_params,
        "parameters_embedding": model.token_embedding.embedding.weight.numel(),
        "architecture": {k: config[k] for k in ("n_layers", "d_model", "n_heads", "d_ff", "seq_len")},
        "vocab_size": tokenizer.vocab_size,
        "training": {
            "steps": step,
            "tokens": step * config["total_batch_tokens"],
            "batch_tokens": config["total_batch_tokens"],
            "lr": config["lr"], "min_lr_ratio": config["min_lr_ratio"],
            "weight_decay": config["weight_decay"], "grad_clip": config["grad_clip"],
            "dataset": manifest.get("dataset"),
            "mean_tokens_per_second": sum(r["tokens_per_s"] for r in trains[1:]) / max(1, len(trains) - 1),
        },
        "final_validation": {"step": vals[-1]["step"], "loss": vals[-1]["val_loss"], "bits_per_byte": vals[-1]["val_bpb"]},
        "best_validation": min(({"step": r["step"], "loss": r["val_loss"], "bits_per_byte": r["val_bpb"]} for r in vals),
                               key=lambda r: r["loss"]),
        "validation_curve": [{"step": r["step"], "loss": r["val_loss"], "bits_per_byte": r["val_bpb"]} for r in vals],
    }
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    fv, bv = summary["final_validation"], summary["best_validation"]
    print(f"Wrote {summary_path}: final val loss {fv['loss']:.4f} ({fv['bits_per_byte']:.4f} bits/byte) at step "
          f"{fv['step']}; best at step {bv['step']}")

    # ---- list of files to back up ----
    files = [(final_path, "final_model.pt"), (summary_path, "summary.json"),
             (latest_path, "full/latest.pt"), (log_path, "logs/log.jsonl"),
             (args.tokenizer, "tokenizer/tokenizer.json"),
             (os.path.join(args.data_dir, "manifest.json"), "data/manifest.json")]
    files += [(p, f"intermediate/{os.path.basename(p)}")
              for p in sorted(glob.glob(os.path.join(args.ckpt_dir, "model_step_*.pt")))]
    if os.path.exists(args.stdout_log):
        files.append((args.stdout_log, "logs/train_stdout.txt"))
    else:
        print(f"Note: {args.stdout_log} not found; the text samples are only in the terminal history")

    print("\nFiles to back up:")
    for local, remote in files:
        print(f"  {remote:<40} {os.path.getsize(local) / 1e6:>9.1f} MB")

    if not args.repo:
        print("\nNo --repo given: nothing uploaded.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, private=True, exist_ok=True)
    existing = set(api.list_repo_files(args.repo))
    for local, remote in files:
        if remote in existing:
            print(f"skip       {remote} (already uploaded)")
            continue
        print(f"uploading  {remote}", flush=True)
        api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=args.repo)

    uploaded = set(api.list_repo_files(args.repo))
    missing = [remote for _, remote in files if remote not in uploaded]
    if missing:
        raise SystemExit(f"Upload incomplete, missing: {missing}. Run the same command again.")
    print(f"\nAll {len(files)} files are in https://huggingface.co/{args.repo} (private)")


if __name__ == "__main__":
    main()
