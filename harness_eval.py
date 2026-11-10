"""
Standard zero-shot benchmarks through EleutherAI's lm-evaluation-harness, for our models and GPT-2.

    pip install lm_eval
    python harness_eval.py                         # both of our models + GPT-2 small/medium/large
    python harness_eval.py --limit 100             # quick test: 100 examples per task
    python harness_eval.py --check-hf              # also run GPT-2 small through the harness's own HF model

The harness supplies the datasets, prompts and metrics; this file only teaches it to
score text with our models. Every task here is multiple choice (or the probability of
a word), so a model never writes free text: answer length and style can't affect scores.

Tasks (zero-shot):
  hellaswag      finish an everyday-activity description (4 endings)
  piqa           physical common sense: pick the better of 2 solutions
  arc_easy       grade-school science questions (easy set)
  arc_challenge  grade-school science questions that simple methods get wrong
  winogrande     resolve a pronoun-like blank (2 options)
  sciq           science exam questions (4 options)
  openbookqa     elementary science questions needing common knowledge
  lambada_openai predict the last word of a passage

acc picks the answer with the highest total log-probability; acc_norm divides it by the
answer's length in bytes (the harness's normalisation, so it differs slightly from the
per-token acc_norm in evaluate.py).

Our model gets [BOS] at the start of every context, as in training. GPT-2 gets no start
token for non-empty contexts, the harness's default for GPT-2; --check-hf confirms our
wrapper then reproduces the harness's own numbers.

Results go to results/harness_results.json and results/harness_report.md, merged by model name.
"""

import argparse
import gc
import json
import os
import sys
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from lm_eval import simple_evaluate
from lm_eval.api.model import TemplateLM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import GPT2Model, OurModel  # noqa: E402

TASKS = ["hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande", "sciq", "openbookqa", "lambada_openai"]
CHANCE = {"hellaswag": 25, "piqa": 50, "arc_easy": 25, "arc_challenge": 25, "winogrande": 50,
          "sciq": 25, "openbookqa": 25, "lambada_openai": 0}


class HarnessModel(TemplateLM):
    """Wraps an evaluate.py model (ours or GPT-2) so the harness can score text with it."""

    def __init__(self, m, device, autocast, use_start, token_budget=16384):
        super().__init__()
        self.m = m
        self.device_ = device
        self.autocast = autocast
        self.use_start = use_start      # prepend the start token to every context
        self.token_budget = token_budget  # max (sequences x length) per forward pass

    @property
    def eot_token_id(self):
        return self.m.stop_id

    @property
    def prefix_token_id(self):
        # Used as the whole context when a request's context is empty.
        return self.m.start_id

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        return self.m.encode(string)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        # requests: [((context, continuation), context_ids, continuation_ids), ...]
        items = []
        for i, (_, ctx, cont) in enumerate(requests):
            whole_text = requests[i][0][0] is not None  # rolling windows (mid-document) get no start token
            if self.use_start and whole_text and (not ctx or ctx[0] != self.m.start_id):
                ctx = [self.m.start_id] + ctx
            seq = (ctx + cont)[-(self.m.max_len + 1):]  # keep the end if it's too long
            items.append((i, seq, len(cont)))
        items.sort(key=lambda t: -len(t[1]))  # longest first: similar lengths share a batch

        out = [None] * len(items)
        pos = 0
        bar = tqdm(total=len(items), disable=disable_tqdm, desc="Scoring")
        while pos < len(items):
            width = len(items[pos][1]) - 1
            n = max(1, self.token_budget // max(width, 1))
            batch = items[pos:pos + n]
            x = torch.zeros(len(batch), width, dtype=torch.long)
            for row, (_, seq, _) in enumerate(batch):
                x[row, :len(seq) - 1] = torch.tensor(seq[:-1])  # right padding is safe with causal attention
            with torch.no_grad(), self.autocast:
                logits = self.m.logits(x.to(self.device_))
            for row, (i, seq, n_cont) in enumerate(batch):
                end = len(seq) - 1
                logp = F.log_softmax(logits[row, end - n_cont:end].float(), dim=-1)
                target = torch.tensor(seq[-n_cont:], device=logp.device)
                total = logp.gather(1, target[:, None]).sum().item()
                greedy = bool((logp.argmax(-1) == target).all())
                out[i] = (total, greedy)
                self.cache_hook.add_partial("loglikelihood", requests[i][0], out[i])
            pos += len(batch)
            bar.update(len(batch))
        bar.close()
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        # Whole-document log-likelihood (perplexity tasks such as wikitext), in disjoint windows.
        results = []
        for (text,) in tqdm([r.args for r in requests], disable=disable_tqdm, desc="Rolling"):
            windows = map(make_disjoint_window, get_rolling_token_windows(
                token_list=self.tok_encode(text), prefix_token=self.prefix_token_id,
                max_seq_len=self.m.max_len, context_len=1))
            scores = self._loglikelihood_tokens([((None, None), c, t) for c, t in windows], disable_tqdm=True)
            results.append(sum(s for s, _ in scores))
        return results

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("Only likelihood-based tasks are supported here.")


def summarize(task_results):
    out = {}
    for task, metrics in task_results.items():
        row = {}
        for key, value in metrics.items():
            if "," in key and isinstance(value, (int, float)):
                name = key.split(",")[0]
                row[name] = value
        out[task] = row
    return out


def write_report(results, path):
    names = sorted(results["models"], key=lambda n: results["models"][n]["parameters"])
    tasks = [t for t in TASKS if any(t in results["models"][n].get("tasks", {}) for n in names)]

    def cell(n, t):
        r = results["models"][n].get("tasks", {}).get(t)
        if not r:
            return ""
        metric = "acc_norm" if "acc_norm" in r else "acc"
        return f"{r[metric] * 100:.1f} ± {r.get(metric + '_stderr', 0) * 100:.1f}"

    dtypes = sorted({results["models"][n].get("dtype", "bfloat16") for n in names})
    lines = ["# Zero-shot benchmarks (lm-evaluation-harness)", "",
             "Accuracy in %, ± one standard error. acc_norm where the task has it (length-normalised by bytes), "
             "otherwise acc. Precision: " + ", ".join(dtypes) + ".", "",
             "| Model | " + " | ".join(tasks) + " | Average |", "|---|" + "---|" * (len(tasks) + 1)]
    for n in names:
        r = results["models"][n].get("tasks", {})
        vals = [(r[t].get("acc_norm", r[t].get("acc"))) * 100 for t in tasks if t in r]
        avg = f"{sum(vals) / len(vals):.1f}" if len(vals) == len(tasks) else ""
        lines.append(f"| {n} | " + " | ".join(cell(n, t) for t in tasks) + f" | {avg} |")
    lines += ["", "Chance: " + ", ".join(f"{t} {CHANCE[t]}%" for t in tasks) + "."]
    if results.get("hf_check"):
        lines += ["", "## Check: our wrapper vs the harness's own Hugging Face model (GPT-2 small)", "",
                  "| Task | Metric | Our wrapper | Harness HF | Difference |", "|---|---|---|---|---|"]
        for row in results["hf_check"].values():
            lines.append(f"| {row['task']} | {row['metric']} | {row['ours'] * 100:.2f} | {row['hf'] * 100:.2f} | "
                         f"{(row['ours'] - row['hf']) * 100:+.2f} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=["ours", "gpt2", "gpt2-medium", "gpt2-large"])
    p.add_argument("--checkpoints", nargs="*", default=["artifacts/final_model.pt", "artifacts_350m/final_model.pt"])
    p.add_argument("--tasks", nargs="*", default=TASKS)
    p.add_argument("--tokenizer", default="Tokenizer/tokenizer.json")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--limit", type=int, default=None, help="Examples per task, for a quick test.")
    p.add_argument("--token-budget", type=int, default=16384, help="Tokens per forward pass (lower if out of memory).")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default=None)
    p.add_argument("--check-hf", action="store_true",
                   help="Also score GPT-2 small with the harness's built-in HF model and compare.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = args.dtype or ("bfloat16" if device.type == "cuda" else "float32")
    autocast = torch.autocast(device.type, dtype=torch.bfloat16) if dtype == "bfloat16" else nullcontext()
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "harness_results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {"models": {}}
    print(f"Device: {device} | dtype: {dtype} | tasks: {', '.join(args.tasks)}")

    jobs = [("ours", c) for c in args.checkpoints] if "ours" in args.models else []
    jobs += [(m, None) for m in args.models if m != "ours"]
    for key, checkpoint in jobs:
        m = OurModel(checkpoint, args.tokenizer, device) if key == "ours" else GPT2Model(key, device)
        print(f"\n=== {m.name} ===")
        lm = HarnessModel(m, device, autocast, use_start=(key == "ours"), token_budget=args.token_budget)
        out = simple_evaluate(model=lm, tasks=args.tasks, num_fewshot=0, limit=args.limit,
                              log_samples=False, bootstrap_iters=1000)
        r = results["models"].setdefault(m.name, {"parameters": m.params})
        r.setdefault("tasks", {}).update(summarize(out["results"]))
        r["limit"] = args.limit
        r["dtype"] = dtype
        for t in args.tasks:
            row = r["tasks"].get(t, {})
            metric = "acc_norm" if "acc_norm" in row else "acc"
            if metric in row:
                print(f"  {t:<15} {metric:<8} {row[metric]:.2%}")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        del lm, m
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.check_hf:
        print("\n=== GPT-2 small through the harness's own HF model (check) ===")
        hf = simple_evaluate(model="hf", model_args="pretrained=openai-community/gpt2",
                             tasks=args.tasks, num_fewshot=0, limit=args.limit, device=str(device),
                             batch_size=32, log_samples=False, bootstrap_iters=1000)
        ref = summarize(hf["results"])
        ours = results["models"].get("GPT-2 small (124M)", {}).get("tasks", {})
        results["hf_check"] = {}
        for t in args.tasks:
            for metric in ("acc", "acc_norm"):
                if metric in ref.get(t, {}) and metric in ours.get(t, {}):
                    results["hf_check"][f"{t}/{metric}"] = {
                        "task": t, "metric": metric, "ours": ours[t][metric], "hf": ref[t][metric]}
                    print(f"  {t:<15} {metric:<8} wrapper {ours[t][metric]:.2%} | harness HF {ref[t][metric]:.2%}")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    write_report(results, os.path.join(args.out_dir, "harness_report.md"))
    print(f"\nWrote {results_path} and {os.path.join(args.out_dir, 'harness_report.md')}")


if __name__ == "__main__":
    main()
