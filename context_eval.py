"""
Do the models use information given in the context? Three tests, same data for every model.

    python context_eval.py                                  # all tests, all models
    python context_eval.py --models ours gpt2 --tests lambada

Tests
  lambada         Predict the last word of a passage, which needs the wider context
                  (LAMBADA, OpenAI version, 5,153 passages). Correct only if greedy
                  decoding produces every token of the word. Also reports its perplexity.
  squad           Reading comprehension: a Wikipedia paragraph, a question, two worked
                  examples first. The answer is generated up to the newline and scored
                  with SQuAD exact match and F1. Each question is also asked without the
                  paragraph ("closed book"), so the gain from the context is visible.
  counterfactual  25 simple fact questions, each asked three ways: no passage, a passage
                  with the true answer, and a passage stating a false answer. Scored by
                  which of the two answers the model finds more likely. Shows how much a
                  model knows, how well it reads, and whether it follows the context or
                  its own memory when they disagree.

Uses the model wrappers from evaluate.py. Results: results/context_results.json and
results/context_report.md.
"""

import argparse
import collections
import json
import os
import re
import string
import time
from contextlib import nullcontext

import torch

from evaluate import GPT2Model, OurModel, masked_loss_sums

LAMBADA = ("EleutherAI/lambada_openai", "data/lambada_test.jsonl")
SQUAD_REPO = "rajpurkar/squad"

# question, true answer, false answer (similar in form, so likelihoods are comparable)
FACTS = [
    ("What is the capital of France?", "Paris", "Lyon"),
    ("What is the capital of Japan?", "Tokyo", "Osaka"),
    ("What is the capital of Italy?", "Rome", "Milan"),
    ("What is the capital of Germany?", "Berlin", "Munich"),
    ("What is the capital of Egypt?", "Cairo", "Alexandria"),
    ("What is the largest planet in the solar system?", "Jupiter", "Saturn"),
    ("Which planet is closest to the Sun?", "Mercury", "Venus"),
    ("Which planet is known as the Red Planet?", "Mars", "Venus"),
    ("What is the chemical symbol for gold?", "Au", "Ag"),
    ("Which gas do plants absorb from the air for photosynthesis?", "carbon dioxide", "nitrogen"),
    ("How many legs does a spider have?", "eight", "six"),
    ("What is the smallest prime number?", "two", "three"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare", "Charles Dickens"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", "Vincent van Gogh"),
    ("What is the largest ocean on Earth?", "the Pacific Ocean", "the Atlantic Ocean"),
    ("What is the longest river in Africa?", "the Nile", "the Congo"),
    ("What is the tallest mountain in the world?", "Mount Everest", "Mount Kilimanjaro"),
    ("What is the hardest natural substance?", "diamond", "quartz"),
    ("Which organ pumps blood through the body?", "the heart", "the liver"),
    ("What is the currency of Japan?", "the yen", "the won"),
    ("What language is mainly spoken in Brazil?", "Portuguese", "Spanish"),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong", "Yuri Gagarin"),
    ("In which year did World War II end?", "1945", "1950"),
    ("At what temperature in Celsius does water boil at sea level?", "100 degrees", "80 degrees"),
    ("Which animal is the largest mammal?", "the blue whale", "the elephant"),
]
FILLER_BEFORE = "The following notes were written for a general knowledge quiz."
FILLER_AFTER = "Students are encouraged to read each note carefully."


# ============================================================
# Helpers
# ============================================================

def download(repo, filename, cache_dir, revision=None):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision, cache_dir=cache_dir)


def squad_validation(cache_dir):
    """Find SQuAD's validation Parquet file wherever the repo currently keeps it."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(SQUAD_REPO, repo_type="dataset")
    candidates = [f for f in files if "validation" in f and f.endswith(".parquet")]
    if candidates:
        path = download(SQUAD_REPO, candidates[0], cache_dir)
    else:  # fall back to Hugging Face's automatic Parquet conversion
        path = download(SQUAD_REPO, "plain_text/validation/0000.parquet", cache_dir, revision="refs/convert/parquet")
    rows = pq.read_table(path).to_pylist()
    return [{"context": r["context"], "question": r["question"], "answers": list(r["answers"]["text"])} for r in rows]


def normalize(text):
    """Official SQuAD answer normalization."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, golds):
    return max(float(normalize(prediction) == normalize(g)) for g in golds)


def f1(prediction, golds):
    def score(pred, gold):
        p, g = normalize(pred).split(), normalize(gold).split()
        common = sum((collections.Counter(p) & collections.Counter(g)).values())
        if common == 0:
            return 0.0
        precision, recall = common / len(p), common / len(g)
        return 2 * precision * recall / (precision + recall)
    return max(score(prediction, g) for g in golds)


@torch.no_grad()
def greedy_until_newline(m, ids, max_new, device, autocast):
    """Greedy generation with the KV cache, stopping at the first newline."""
    out, cache = [], None
    with autocast:
        logits, cache = m.step(torch.tensor([ids], device=device), cache)
        for _ in range(max_new):
            next_id = int(logits[0, -1].float().argmax())
            if next_id == m.stop_id:
                break
            out.append(next_id)
            if "\n" in m.decode(out):
                break
            if len(ids) + len(out) >= m.max_len:
                break
            logits, cache = m.step(torch.tensor([[next_id]], device=device), cache)
    return m.decode(out).split("\n")[0].strip()


def prefix(m, use_start):
    return [m.start_id] if use_start else []


# ============================================================
# Tests
# ============================================================

def test_lambada(m, passages, device, autocast, use_start):
    correct, loss_sum, n_tokens = 0, 0.0, 0
    for text in passages:
        context, word = text.rsplit(" ", 1)
        target = m.encode(" " + word)
        ctx_ids = prefix(m, use_start) + m.encode(context)
        ctx_ids = ctx_ids[-(m.max_len + 1 - len(target)):]  # keep the most recent context if it's too long
        ids = torch.tensor([ctx_ids + target], device=device)
        with torch.no_grad(), autocast:
            logits = m.logits(ids[:, :-1])[0].float()
        pred = logits[len(ctx_ids) - 1:].argmax(-1)
        tgt = ids[0, len(ctx_ids):]
        correct += bool((pred == tgt).all())
        loss_sum += torch.nn.functional.cross_entropy(logits[len(ctx_ids) - 1:], tgt, reduction="sum").item()
        n_tokens += len(target)
    return {"examples": len(passages), "accuracy": correct / len(passages),
            "word_perplexity": float(torch.exp(torch.tensor(loss_sum / len(passages))))}


def squad_prompt(demos, ex, with_context):
    blocks = []
    for d in demos + [ex]:
        answer = f" {d['answers'][0]}" if d is not ex else ""
        if with_context:
            blocks.append(f"{d['context']}\nQuestion: {d['question']}\nAnswer:{answer}")
        else:
            blocks.append(f"Question: {d['question']}\nAnswer:{answer}")
    return "\n\n".join(blocks)


def test_squad(m, demos, examples, device, autocast):
    results = {}
    for mode, with_context in (("open_book", True), ("closed_book", False)):
        em = f1_total = 0.0
        shown = []
        for ex in examples:
            used = list(demos)
            ids = [m.start_id] + m.encode(squad_prompt(used, ex, with_context))
            while len(ids) > m.max_len - 24 and used:  # drop worked examples if the prompt is too long
                used = used[:-1]
                ids = [m.start_id] + m.encode(squad_prompt(used, ex, with_context))
            ids = ids[-(m.max_len - 24):]
            answer = greedy_until_newline(m, ids, 20, device, autocast)
            em += exact_match(answer, ex["answers"])
            f1_total += f1(answer, ex["answers"])
            if len(shown) < 5:
                shown.append({"question": ex["question"], "gold": ex["answers"][0], "answer": answer})
        results[mode] = {"exact_match": em / len(examples), "f1": f1_total / len(examples), "samples": shown}
    results["examples"] = len(examples)
    results["f1_gain_from_context"] = results["open_book"]["f1"] - results["closed_book"]["f1"]
    return results


def answer_preference(m, prompt, true, false, device, autocast, use_start):
    """True if the model gives the true answer a higher total log-likelihood than the false one."""
    answers = [m.encode(" " + true), m.encode(" " + false)]
    base = prefix(m, use_start) + m.encode(prompt)
    base = base[-(m.max_len + 1 - max(map(len, answers))):]
    seqs = [base + a for a in answers]
    loss, _ = masked_loss_sums(m, seqs, [len(base) - 1] * 2, device, autocast)
    return bool(loss[0] < loss[1])


def test_counterfactual(m, device, autocast, use_start):
    counts = {"closed_book": 0, "supporting_passage": 0, "counterfactual_follows_context": 0}
    rows = []
    for question, true, false in FACTS:
        statement = f"Note: The answer to the question \"{question}\" is {{}}."
        closed = f"Question: {question}\nAnswer:"
        support = f"{FILLER_BEFORE}\n{statement.format(true)}\n{FILLER_AFTER}\n\nQuestion: {question}\nAnswer:"
        counter = f"{FILLER_BEFORE}\n{statement.format(false)}\n{FILLER_AFTER}\n\nQuestion: {question}\nAnswer:"
        knows = answer_preference(m, closed, true, false, device, autocast, use_start)
        reads = answer_preference(m, support, true, false, device, autocast, use_start)
        follows = not answer_preference(m, counter, true, false, device, autocast, use_start)
        counts["closed_book"] += knows
        counts["supporting_passage"] += reads
        counts["counterfactual_follows_context"] += follows
        rows.append({"question": question, "true": true, "false": false,
                     "knows": knows, "reads_support": reads, "follows_counterfactual": follows})
    n = len(FACTS)
    return {"facts": n, **{k: v / n for k, v in counts.items()}, "per_fact": rows}


# ============================================================
# Report
# ============================================================

def write_report(results, path):
    names = list(results["models"])
    sections = ["# Using context: our model vs GPT-2"]

    def table(header, rows):
        return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                         ["| " + " | ".join(r) + " |" for r in rows])

    rows = [(n, r["lambada"]) for n in names if "lambada" in (r := results["models"][n])]
    if rows:
        sections += ["## LAMBADA (predict the last word from the passage)",
                     table(["Model", "Accuracy", "Perplexity of the target word"],
                           [[n, f"{r['accuracy']:.2%}", f"{r['word_perplexity']:.1f}"] for n, r in rows])]

    rows = [(n, r["squad"]) for n in names if "squad" in (r := results["models"][n])]
    if rows:
        sections += ["## SQuAD reading comprehension (2 worked examples)",
                     table(["Model", "With passage: EM", "With passage: F1", "Without passage: EM",
                            "Without passage: F1", "F1 gain from passage"],
                           [[n, f"{r['open_book']['exact_match']:.1%}", f"{r['open_book']['f1']:.1%}",
                             f"{r['closed_book']['exact_match']:.1%}", f"{r['closed_book']['f1']:.1%}",
                             f"{r['f1_gain_from_context'] * 100:+.1f} pts"] for n, r in rows]),
                     "Sample answers (with passage):\n\n" + "\n".join(
                         f"- **{n}**: " + "; ".join(f"Q: {s['question']} → *{s['answer']}* (gold: {s['gold']})"
                                                    for s in r["open_book"]["samples"][:3]) for n, r in rows)]

    rows = [(n, r["counterfactual"]) for n in names if "counterfactual" in (r := results["models"][n])]
    if rows:
        sections += ["## Closed book vs open book vs counterfactual passage",
                     "Share of the 25 fact questions where the model prefers the true answer (first two columns), "
                     "or prefers the passage's false answer (last column). 50% is chance.",
                     table(["Model", "Knows it (no passage)", "Reads it (true passage)", "Follows a false passage"],
                           [[n, f"{r['closed_book']:.0%}", f"{r['supporting_passage']:.0%}",
                             f"{r['counterfactual_follows_context']:.0%}"] for n, r in rows])]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sections) + "\n")


# ============================================================
# Main
# ============================================================

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=["ours", "gpt2", "gpt2-medium", "gpt2-large"])
    p.add_argument("--tests", nargs="*", default=["lambada", "squad", "counterfactual"])
    p.add_argument("--checkpoint", default="artifacts/final_model.pt")
    p.add_argument("--tokenizer", default="Tokenizer/tokenizer.json")
    p.add_argument("--cache-dir", default="eval_data")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--squad-examples", type=int, default=1000)
    p.add_argument("--lambada-limit", type=int, default=None)
    p.add_argument("--lambada-file", help="Local LAMBADA jsonl ({'text': ...} per line).")
    p.add_argument("--squad-file", help="Local SQuAD jsonl ({'context','question','answers'} per line).")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default=None)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = args.dtype or ("bfloat16" if device.type == "cuda" else "float32")
    autocast = torch.autocast(device.type, dtype=torch.bfloat16) if dtype == "bfloat16" else nullcontext()
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "context_results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {"models": {}}

    passages = demos = squad = None
    if "lambada" in args.tests:
        path = args.lambada_file or download(*LAMBADA, args.cache_dir)
        passages = [json.loads(line)["text"] for line in open(path, encoding="utf-8")]
        passages = passages[:args.lambada_limit] if args.lambada_limit else passages
        print(f"LAMBADA: {len(passages):,} passages")
    if "squad" in args.tests:
        rows = ([json.loads(l) for l in open(args.squad_file, encoding="utf-8")] if args.squad_file
                else squad_validation(args.cache_dir))
        # Two short worked examples, then evaluation questions from different paragraphs.
        by_length = sorted(range(len(rows)), key=lambda i: len(rows[i]["context"]))
        demo_idx = [by_length[0], next(i for i in by_length[1:] if rows[i]["context"] != rows[by_length[0]]["context"])]
        demos = [rows[i] for i in demo_idx]
        demo_contexts = {d["context"] for d in demos}
        squad = [r for r in rows if r["context"] not in demo_contexts][::max(1, len(rows) // args.squad_examples)]
        squad = squad[:args.squad_examples]
        print(f"SQuAD: {len(squad):,} questions, 2 worked examples")

    for key in args.models:
        print(f"\n=== {key} ===")
        m = OurModel(args.checkpoint, args.tokenizer, device) if key == "ours" else GPT2Model(key, device)
        r = results["models"].setdefault(m.name, {})
        use_start = key == "ours"  # GPT-2 without a start token, as in its published evaluations
        start = time.time()
        if passages:
            r["lambada"] = test_lambada(m, passages, device, autocast, use_start)
            print(f"  LAMBADA accuracy {r['lambada']['accuracy']:.2%} | word perplexity {r['lambada']['word_perplexity']:.1f}")
        if squad:
            r["squad"] = test_squad(m, demos, squad, device, autocast)
            s = r["squad"]
            print(f"  SQuAD with passage EM {s['open_book']['exact_match']:.1%} F1 {s['open_book']['f1']:.1%} | "
                  f"without EM {s['closed_book']['exact_match']:.1%} F1 {s['closed_book']['f1']:.1%}")
        if "counterfactual" in args.tests:
            r["counterfactual"] = test_counterfactual(m, device, autocast, use_start)
            c = r["counterfactual"]
            print(f"  facts: knows {c['closed_book']:.0%} | reads {c['supporting_passage']:.0%} | "
                  f"follows false passage {c['counterfactual_follows_context']:.0%}")
        print(f"  ({time.time() - start:.0f}s)")

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = os.path.join(args.out_dir, "context_report.md")
    write_report(results, report)
    print(f"\nWrote {results_path} and {report}")


if __name__ == "__main__":
    main()
