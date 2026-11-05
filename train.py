"""
Pre-train the decoder-only transformer on the tokenized FineWeb-Edu shards.

Single GPU:
    python train.py --data-dir data --tokenizer Tokenizer/tokenizer.json

Both GPUs (distributed data parallel):
    torchrun --standalone --nproc_per_node=2 train.py --data-dir data --tokenizer Tokenizer/tokenizer.json

Measure speed first (runs a few steps, prints tokens/s and the estimated total time):
    torchrun --standalone --nproc_per_node=2 train.py --benchmark-steps 30 ...

Resume after an interruption (automatic if out-dir/latest.pt exists):
    the same command again. Ctrl+C (or SIGTERM) saves a checkpoint and exits cleanly.

Data loading is a deterministic function of the step number, so a resumed run
sees exactly the batches it would have seen without the interruption.
"""

import argparse
import json
import math
import os
import signal
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Tokenizer"))

from Tokenizer.byte_level_bpe import ByteLevelBPETokenizer  # noqa: E402
from Decoder_Blocks.transformer_model import TransformerModel  # noqa: E402


# ============================================================
# Arguments
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    # data and output
    p.add_argument("--data-dir", default="data")
    p.add_argument("--tokenizer", default="Tokenizer/tokenizer.json")
    p.add_argument("--out-dir", default="checkpoints")
    # model (defaults: GPT-2 small shape)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--seq-len", type=int, default=1024)
    # batch and schedule
    p.add_argument("--micro-batch", type=int, default=16, help="Sequences per GPU per forward pass.")
    p.add_argument("--total-batch-tokens", type=int, default=524_288, help="Tokens per optimizer step (2^19).")
    p.add_argument("--max-tokens", type=float, default=2.5e9, help="Total training tokens.")
    p.add_argument("--max-steps", type=int, default=None, help="Override the step count derived from max-tokens.")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=None, help="Default: 3%% of training steps.")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # evaluation, logging, checkpoints
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=64,
                   help="Validation micro-batches per eval, in total across all GPUs (must divide by the GPU count).")
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=250)
    p.add_argument("--keep-every", type=int, default=1000, help="Also keep a weights-only copy every N steps.")
    p.add_argument("--no-resume", action="store_true", help="Start fresh even if a checkpoint exists.")
    # system
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default=None,
                   help="Default: bfloat16 on GPU, float32 on CPU.")
    p.add_argument("--compile", action="store_true", help="Use torch.compile for the training model.")
    p.add_argument("--peak-tflops", type=float, default=165.0, help="Per-GPU peak for MFU (A30 bf16: 165).")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--benchmark-steps", type=int, default=0,
                   help="Run this many steps, report speed and estimated total time, then exit.")
    return p.parse_args(argv)


# ============================================================
# Distributed setup
# ============================================================

def setup_distributed():
    if "RANK" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return rank, world, device, True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device, False


def all_reduce_sum(tensor, distributed):
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


# ============================================================
# Data
# ============================================================

class ShardStream:
    """Fixed-size chunks of the token stream, as a pure function of the micro-step.

    The shards are treated as one long stream (in a shuffled order per epoch for
    training). Chunk c holds tokens [c * B*T, c * B*T + B*T + 1), giving inputs
    and targets offset by one. Rank r at micro-step k reads chunk k * world + r,
    so the GPUs never read the same data and a resumed run continues exactly.
    """

    def __init__(self, files, micro_batch, seq_len, rank, world, seed, shuffle):
        self.shards = [np.load(f, mmap_mode="r") for f in files]
        self.B, self.T = micro_batch, seq_len
        self.rank, self.world = rank, world
        self.seed, self.shuffle = seed, shuffle
        self.chunk = micro_batch * seq_len
        total = sum(len(s) for s in self.shards)
        self.chunks_per_epoch = (total - 1) // self.chunk
        if self.chunks_per_epoch < 1:
            raise ValueError("Not enough tokens for a single micro-batch.")
        self._epoch = None

    def _layout(self, epoch):
        if self._epoch != epoch:
            order = list(range(len(self.shards)))
            if self.shuffle:
                order = list(np.random.default_rng(self.seed + epoch).permutation(len(self.shards)))
            starts = np.cumsum([0] + [len(self.shards[i]) for i in order])
            self._order, self._starts, self._epoch = order, starts, epoch
        return self._order, self._starts

    def _read(self, epoch, start, length):
        order, starts = self._layout(epoch)
        out = np.empty(length, dtype=np.int64)
        filled = 0
        i = int(np.searchsorted(starts, start, side="right") - 1)
        offset = start - starts[i]
        while filled < length:
            shard = self.shards[order[i]]
            take = min(length - filled, len(shard) - offset)
            out[filled:filled + take] = shard[offset:offset + take]
            filled += take
            i, offset = i + 1, 0
        return out

    def batch(self, micro_step):
        c = micro_step * self.world + self.rank
        epoch, index = divmod(c, self.chunks_per_epoch)
        tokens = torch.from_numpy(self._read(epoch, index * self.chunk, self.chunk + 1))
        x = tokens[:-1].view(self.B, self.T)
        y = tokens[1:].view(self.B, self.T)
        return x, y


def shard_paths(data_dir, manifest, key):
    # Manifest paths come from the machine that tokenized; resolve them inside data_dir.
    return [os.path.join(data_dir, os.path.basename(p)) for p in manifest[key]]


# ============================================================
# Learning rate
# ============================================================

def learning_rate(step, max_steps, warmup, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================
# Evaluation and sampling
# ============================================================

@torch.no_grad()
def evaluate(model, val_stream, n_batches, token_bytes, device, autocast, distributed, vocab_size):
    model.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)  # loss sum, tokens, bytes
    for k in range(n_batches):
        x, y = val_stream.batch(k)
        x, y = x.to(device), y.to(device)
        with autocast:
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, vocab_size), y.view(-1), reduction="sum")
        totals[0] += loss.double()
        totals[1] += y.numel()
        totals[2] += token_bytes[y].sum().double()
    all_reduce_sum(totals, distributed)
    model.train()
    loss = (totals[0] / totals[1]).item()
    bits_per_byte = (totals[0] / math.log(2) / totals[2]).item()
    return loss, bits_per_byte


@torch.no_grad()
def generate(model, tokenizer, prompt, device, autocast, max_new_tokens, max_len, temperature=0.8, top_k=50):
    model.eval()
    ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
    x = torch.tensor([ids], device=device)
    out = list(ids)
    with autocast:
        logits, cache = model(x, use_cache=True)
        for _ in range(max_new_tokens):
            if len(out) >= max_len:
                break
            next_logits = logits[0, -1].float() / temperature
            values, indices = torch.topk(next_logits, top_k)
            probs = torch.softmax(values, dim=-1)
            next_id = indices[torch.multinomial(probs, 1)].item()
            out.append(next_id)
            if next_id == tokenizer.eos_id:
                break
            logits, cache = model(torch.tensor([[next_id]], device=device), past_kv=cache, use_cache=True)
    model.train()
    return tokenizer.decode(out)


# ============================================================
# Checkpoints
# ============================================================

def save_checkpoint(path, model, optimizer, step, args, extra):
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": vars(args),
        **extra,
    }, tmp)
    os.replace(tmp, path)  # never leaves a half-written checkpoint behind


MUST_MATCH = ["n_layers", "d_model", "n_heads", "d_ff", "seq_len", "total_batch_tokens", "micro_batch"]


# ============================================================
# Main
# ============================================================

def main(argv=None):
    args = parse_args(argv)
    rank, world, device, distributed = setup_distributed()
    master = rank == 0
    log = print if master else (lambda *a, **k: None)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype_name = args.dtype or ("bfloat16" if device.type == "cuda" else "float32")
    autocast = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if dtype_name == "bfloat16" else nullcontext())

    # ---- tokenizer and data ----
    tokenizer = ByteLevelBPETokenizer.load(args.tokenizer)
    with open(os.path.join(args.data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    if manifest["vocab_size"] != tokenizer.vocab_size or manifest["tokenizer_merges"] != len(tokenizer.merges):
        raise ValueError("The tokenizer does not match the one used to build the shards.")
    vocab_size = tokenizer.vocab_size

    tokens_per_micro = args.micro_batch * args.seq_len * world
    if args.total_batch_tokens % tokens_per_micro != 0:
        raise ValueError(f"total-batch-tokens must be divisible by micro_batch * seq_len * GPUs = {tokens_per_micro}")
    grad_accum = args.total_batch_tokens // tokens_per_micro
    if args.eval_batches % world != 0:
        raise ValueError(f"eval-batches must be divisible by the number of GPUs ({world})")
    eval_batches_per_gpu = args.eval_batches // world

    train_stream = ShardStream(shard_paths(args.data_dir, manifest, "train_files"),
                               args.micro_batch, args.seq_len, rank, world, args.seed, shuffle=True)
    val_stream = ShardStream(shard_paths(args.data_dir, manifest, "val_files"),
                             args.micro_batch, args.seq_len, rank, world, args.seed, shuffle=False)

    # Bytes each token represents, for bits-per-byte (special tokens count as 0 bytes).
    token_bytes = torch.tensor([0 if i in tokenizer.special_ids else len(tokenizer.itos[i])
                                for i in range(vocab_size)], device=device)

    max_steps = args.max_steps or int(args.max_tokens // args.total_batch_tokens)
    warmup = args.warmup_steps if args.warmup_steps is not None else max(1, int(0.03 * max_steps))
    min_lr = args.lr * args.min_lr_ratio

    # ---- model ----
    raw_model = TransformerModel(vocab_size=vocab_size, d_model=args.d_model, padding_id=tokenizer.pad_id,
                                 num_heads=args.n_heads, d_ff=args.d_ff, max_seq_len=args.seq_len,
                                 num_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())  # tied weights are counted once

    decay = [p for p in raw_model.parameters() if p.dim() >= 2]
    no_decay = [p for p in raw_model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=(device.type == "cuda"))

    # ---- resume ----
    os.makedirs(args.out_dir, exist_ok=True)
    latest = os.path.join(args.out_dir, "latest.pt")
    start_step, best_val = 0, float("inf")
    if os.path.exists(latest) and not args.no_resume and not args.benchmark_steps:
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        for key in MUST_MATCH:
            if ckpt["config"][key] != getattr(args, key):
                raise ValueError(f"Checkpoint has {key}={ckpt['config'][key]}, but this run uses {getattr(args, key)}")
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val = ckpt["step"], ckpt.get("best_val", float("inf"))
        log(f"Resumed from {latest} at step {start_step}")
        del ckpt

    model = torch.compile(raw_model) if args.compile else raw_model
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    # FLOPs per token: 6N for the weights plus attention scores and values.
    flops_per_token = 6 * n_params + 12 * args.n_layers * args.d_model * args.seq_len

    log(f"Parameters: {n_params / 1e6:.1f}M | GPUs: {world} | dtype: {dtype_name} | compile: {args.compile}")
    log(f"Batch: {args.total_batch_tokens:,} tokens/step = {args.micro_batch} seq x {args.seq_len} x "
        f"{world} GPUs x {grad_accum} accumulation steps")
    log(f"Steps: {max_steps:,} ({max_steps * args.total_batch_tokens / 1e9:.2f}B tokens), warmup {warmup}, "
        f"lr {args.lr} -> {min_lr}")

    # ---- clean shutdown on Ctrl+C / SIGTERM: finish the step, save, exit ----
    stop_requested = {"flag": False}

    def request_stop(signum, frame):
        stop_requested["flag"] = True
        log("\nStop requested: saving a checkpoint after this step...")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    log_path = os.path.join(args.out_dir, "log.jsonl")
    end_step = start_step + args.benchmark_steps if args.benchmark_steps else max_steps
    bench_times = []

    def checkpoint(step):
        if master:
            save_checkpoint(latest, raw_model, optimizer, step, args, {"best_val": best_val})

    model.train()
    for step in range(start_step, end_step):
        t0 = time.time()
        lr = learning_rate(step, max_steps, warmup, args.lr, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)

        for micro in range(grad_accum):
            x, y = train_stream.batch(step * grad_accum + micro)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if distributed:
                # Only synchronize gradients across GPUs on the last micro-step.
                model.require_backward_grad_sync = (micro == grad_accum - 1)
            with autocast:
                logits = model(x)
            loss = F.cross_entropy(logits.float().view(-1, vocab_size), y.view(-1),
                                   ignore_index=tokenizer.pad_id) / grad_accum
            loss.backward()
            loss_sum += loss.detach()

        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        all_reduce_sum(loss_sum, distributed)
        train_loss = loss_sum.item() / world
        dt = time.time() - t0
        tokens_per_s = args.total_batch_tokens / dt
        done = step + 1

        if not math.isfinite(train_loss):
            raise RuntimeError(f"Loss became {train_loss} at step {step}")

        if args.benchmark_steps:
            bench_times.append(dt)
            log(f"bench step {done - start_step}/{args.benchmark_steps} | loss {train_loss:.4f} | "
                f"{dt * 1000:.0f} ms | {tokens_per_s:,.0f} tok/s")
        elif done % args.log_every == 0 or done == start_step + 1:
            mfu = tokens_per_s * flops_per_token / (world * args.peak_tflops * 1e12)
            eta_h = (max_steps - done) * dt / 3600
            log(f"step {done:>6}/{max_steps} | loss {train_loss:.4f} | lr {lr:.2e} | grad {grad_norm:.2f} | "
                f"{dt * 1000:.0f} ms | {tokens_per_s:,.0f} tok/s | MFU {mfu:.1%} | ETA {eta_h:.1f} h")
            if master:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": done, "train_loss": train_loss, "lr": lr,
                                        "grad_norm": float(grad_norm), "tokens_per_s": tokens_per_s}) + "\n")

        if not args.benchmark_steps:
            if done % args.eval_every == 0 or done == max_steps:
                val_loss, bpb = evaluate(model, val_stream, eval_batches_per_gpu, token_bytes, device,
                                         autocast, distributed, vocab_size)
                best_val = min(best_val, val_loss)
                log(f"  validation | loss {val_loss:.4f} | bits/byte {bpb:.4f} | best {best_val:.4f}")
                if master:
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"step": done, "val_loss": val_loss, "val_bpb": bpb}) + "\n")

            if master and (done % args.sample_every == 0 or done == max_steps):
                text = generate(raw_model, tokenizer, "The most important thing about learning is", device,
                                autocast, max_new_tokens=60, max_len=args.seq_len)
                log(f"  sample | {text!r}")

            if done % args.ckpt_every == 0 or done == max_steps:
                checkpoint(done)
            if master and done % args.keep_every == 0:
                torch.save({"model": raw_model.state_dict(), "step": done, "config": vars(args)},
                           os.path.join(args.out_dir, f"model_step_{done:06d}.pt"))

        # Stop together: if any process was asked to stop, all of them save and exit.
        flag = torch.tensor(float(stop_requested["flag"]), device=device)
        all_reduce_sum(flag, distributed)
        if flag.item() > 0:
            if not args.benchmark_steps:
                checkpoint(done)
                log(f"Saved checkpoint at step {done}. Run the same command to resume.")
            break

    if args.benchmark_steps and bench_times:
        timed = bench_times[3:] if len(bench_times) > 5 else bench_times  # skip warm-up steps
        step_time = sum(timed) / len(timed)
        tok_s = args.total_batch_tokens / step_time
        mfu = tok_s * flops_per_token / (world * args.peak_tflops * 1e12)
        log(f"\nBenchmark: {step_time:.2f} s/step, {tok_s:,.0f} tokens/s, MFU {mfu:.1%}")
        log(f"Full run: {max_steps:,} steps -> about {max_steps * step_time / 3600:.1f} hours")
        if device.type == "cuda":
            log(f"Peak GPU memory: {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB per GPU")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()