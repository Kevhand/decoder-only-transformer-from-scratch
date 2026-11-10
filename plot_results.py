"""
Plot the training runs and the comparisons with GPT-2, as PNGs for the write-up.

    pip install matplotlib
    python plot_results.py

By default it reads both runs (110M in checkpoints/, 336M in checkpoints_350m/);
pick others with --run LABEL=PATH (repeatable). Also reads results/eval_results.json
(evaluate.py), results/context_results.json (context_eval.py) and
results/domain_results.json (domain_eval.py) and results/harness_results.json
(harness_eval.py) if they exist.
Writes:
    results/training_curves.png     loss, validation bits/byte, learning rate, gradient norm vs tokens
    results/model_comparison.png    bits/byte and HellaSwag for our models and GPT-2 small/medium/large
    results/context_comparison.png  SQuAD F1 with vs without the passage, and LAMBADA accuracy
    results/domain_comparison.png   bits/byte on FineWeb-Edu, Wikipedia, books and general web (domain_eval.py)
    results/benchmark_comparison.png  zero-shot benchmarks through lm-evaluation-harness (harness_eval.py)
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Colours by role (reference data-viz palette, light mode)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8984"
GRID = "#e6e5e1"
OURS = "#2a78d6"       # our models: one hue, the smaller run lighter
OURS_LIGHT = "#8ab4e8"
REFERENCE = "#9b9a94"  # GPT-2 reference lines and bars


def style_axes(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)


def billions(x, _):
    return f"{x:.0f}B" if x == int(x) else f"{x:.1f}B"


def smooth(values, window):
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def label_end(ax, x, y, text, color, longest):
    # The longest run: right-aligned under its last point, so it never runs past the plot area.
    # Shorter runs: just right of their last point, in the space the longer run leaves free.
    if longest:
        ax.annotate(text, (x, y), xytext=(0, -9), textcoords="offset points", fontsize=8.5,
                    color=color, va="top", ha="right", fontweight="bold")
    else:
        ax.annotate(text, (x, y), xytext=(7, 0), textcoords="offset points", fontsize=8.5,
                    color=color, va="center", ha="left", fontweight="bold")


def run_colors(n):
    # Smaller runs lighter, the largest run in the full accent.
    return [OURS_LIGHT] * (n - 1) + [OURS] if n > 1 else [OURS]


# ------------------------------------------------------------
# 1. Training curves, all runs on one set of axes
# ------------------------------------------------------------

def training_curves(runs, batch_tokens, gpt2, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE, layout="constrained")
    (a, b), (c, d) = axes
    tok = lambda rs: [r["step"] * batch_tokens / 1e9 for r in rs]  # noqa: E731
    lowest_loss, lowest_bpb, top_grad = [], [], []
    last_step = max(records[-1]["step"] for _, records in runs)

    for (label, records), color in zip(runs, run_colors(len(runs))):
        train = [r for r in records if "train_loss" in r]
        val = [r for r in records if "val_loss" in r]
        t_x, v_x = tok(train), tok(val)
        longest = records[-1]["step"] == last_step

        # Loss: smoothed training loss as a line, validation as dots
        t_s = smooth([r["train_loss"] for r in train], 10)
        v_y = [r["val_loss"] for r in val]
        a.plot(t_x, t_s, color=color, linewidth=1.6, alpha=0.6)
        a.plot(v_x, v_y, color=color, linewidth=0, marker="o", markersize=3.5,
               markeredgecolor=SURFACE, markeredgewidth=1)
        label_end(a, v_x[-1], v_y[-1], f"{label}  {v_y[-1]:.3f}", color, longest)
        lowest_loss += v_y + t_s[len(t_s) // 10:]

        # Validation bits per byte
        bpb = [r["val_bpb"] for r in val]
        b.plot(v_x, bpb, color=color, linewidth=2, marker="o", markersize=3.5,
               markeredgecolor=SURFACE, markeredgewidth=1)
        label_end(b, v_x[-1], bpb[-1], f"{label}  {bpb[-1]:.3f}", color, longest)
        lowest_bpb += bpb

        # Learning rate and gradient norm
        c.plot(t_x, [r["lr"] for r in train], color=color, linewidth=2, label=label)
        g = smooth([r["grad_norm"] for r in train], 10)
        d.plot(t_x, g, color=color, linewidth=1.6, label=label)
        top_grad += g[len(g) // 20:]

    a.set_ylim(min(lowest_loss) - 0.35, 7)  # start the axis after the first drop
    style_axes(a, "Loss (line: training, smoothed; dots: validation)", "Cross-entropy (nats/token)")

    for name, value in gpt2.items():
        b.axhline(value, color=REFERENCE, linewidth=1, linestyle=(0, (4, 3)))
        # At the left edge, where the curves are still high above the reference lines
        b.annotate(f"{name} {value:.3f}", (0.02, value), xycoords=("axes fraction", "data"),
                   xytext=(0, 3), textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
    b.set_ylim(min(lowest_bpb + list(gpt2.values())) - 0.05, 1.6)
    style_axes(b, "Validation bits per byte (dashed: GPT-2 on held-out text)", "Bits per byte (lower is better)")

    style_axes(c, "Learning rate", "Learning rate")
    c.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}" if v else "0"))
    c.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper right")

    d.set_ylim(0, max(top_grad) * 1.6)
    style_axes(d, "Gradient norm (before clipping at 1.0)", "L2 norm, smoothed")
    d.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper right")

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(FuncFormatter(billions))
        ax.set_xlabel("Training tokens", fontsize=9, color=TEXT_SECONDARY)

    sizes = ", ".join(f"{label} on {records[-1]['step'] * batch_tokens / 1e9:.1f}B tokens" for label, records in runs)
    fig.suptitle(f"Decoder-only transformers trained from scratch on FineWeb-Edu ({sizes})",
                 x=0.01, ha="left", fontsize=13, color=TEXT, fontweight="bold")
    fig.get_layout_engine().set(h_pad=0.15, w_pad=0.2, hspace=0.08, wspace=0.06)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ------------------------------------------------------------
# 2. Bits/byte and HellaSwag, our models vs GPT-2
# ------------------------------------------------------------

def bar_colors(names, params):
    ours = sorted((params[n] for n in names if n.startswith("Ours")))
    shade = dict(zip(ours, run_colors(len(ours))))
    return [shade[params[n]] if n.startswith("Ours") else REFERENCE for n in names]


def model_comparison(results, path):
    models = results["models"]
    names = sorted((n for n, r in models.items() if "bpb" in r and "hellaswag" in r),
                   key=lambda n: models[n]["parameters"])
    params = {n: models[n]["parameters"] for n in names}
    colors = bar_colors(names, params)
    y = list(range(len(names)))

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE, layout="constrained")
    for ax, values, title, xlabel, fmt in (
            (a, [models[n]["bpb"]["bits_per_byte"] for n in names],
             "Bits per byte on held-out FineWeb-Edu", "Lower is better", "{:.3f}"),
            (b, [models[n]["hellaswag"]["acc_norm"] * 100 for n in names],
             "HellaSwag accuracy (acc_norm)", "Higher is better", "{:.1f}%")):
        ax.barh(y, values, color=colors, height=0.55)
        for i, v in enumerate(values):
            ax.annotate(fmt.format(v), (v, i), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        style_axes(ax, title, "")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
        ax.set_xlim(0, max(values) * 1.18)

    b.axvline(25, color=TEXT_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    b.annotate("random 25%", (25, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -4),
               textcoords="offset points", fontsize=8, color=TEXT_SECONDARY, va="top", annotation_clip=False)
    b.set_yticklabels([])

    fig.suptitle("Our models vs GPT-2, evaluated on identical data",
                 x=0.01, ha="left", fontsize=13, color=TEXT, fontweight="bold")
    fig.get_layout_engine().set(w_pad=0.2, wspace=0.08)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ------------------------------------------------------------
# 3. Using context: SQuAD with vs without the passage, LAMBADA
# ------------------------------------------------------------

def context_comparison(context, params, path):
    models = context["models"]
    names = sorted((n for n, r in models.items() if "squad" in r and "lambada" in r),
                   key=lambda n: params.get(n, 0))
    if not names:
        return False
    colors = bar_colors(names, {n: params.get(n, 0) for n in names})
    y = list(range(len(names)))
    h = 0.36

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE, layout="constrained")

    # SQuAD F1: solid bar with the passage, pale bar without it
    open_f1 = [models[n]["squad"]["open_book"]["f1"] * 100 for n in names]
    closed_f1 = [models[n]["squad"]["closed_book"]["f1"] * 100 for n in names]
    a.barh([i - h / 2 for i in y], open_f1, color=colors, height=h)
    a.barh([i + h / 2 for i in y], closed_f1, color=colors, alpha=0.3, height=h)
    for i in y:
        a.annotate(f"{open_f1[i]:.1f}", (open_f1[i], i - h / 2), xytext=(4, 0), textcoords="offset points",
                   va="center", fontsize=8.5, color=TEXT)
        a.annotate(f"{closed_f1[i]:.1f}", (closed_f1[i], i + h / 2), xytext=(4, 0), textcoords="offset points",
                   va="center", fontsize=8.5, color=TEXT_SECONDARY)
    a.set_xlim(0, max(open_f1) * 1.18)

    # LAMBADA last-word accuracy
    lam = [models[n]["lambada"]["accuracy"] * 100 for n in names]
    b.barh(y, lam, color=colors, height=0.55)
    for i, v in enumerate(lam):
        b.annotate(f"{v:.1f}%", (v, i), xytext=(4, 0), textcoords="offset points",
                   va="center", fontsize=9, color=TEXT)
    b.set_xlim(0, max(lam) * 1.18)

    for ax, title, xlabel in ((a, "SQuAD F1, 2-shot", "F1 (%)   solid: with the passage, pale: question only"),
                              (b, "LAMBADA: last word correct", "Accuracy (%)")):
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        style_axes(ax, title, "")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    b.set_yticklabels([])

    fig.suptitle("Using context: reading a passage vs answering from memory",
                 x=0.01, ha="left", fontsize=13, color=TEXT, fontweight="bold")
    fig.get_layout_engine().set(w_pad=0.2, wspace=0.08)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return True


# ------------------------------------------------------------
# 4. Bits per byte by kind of text (domain_eval.py)
# ------------------------------------------------------------

DOMAIN_TITLES = {"fineweb_edu": "FineWeb-Edu (training domain)", "wiki": "Wikipedia (WikiText-103)",
                 "books": "Books (PG-19)", "web": "General web (C4)"}


def domain_comparison(domain_results, path):
    models = domain_results["models"]
    domains = [d for d in DOMAIN_TITLES if d in domain_results["domains"]]
    names = sorted(models, key=lambda n: models[n]["parameters"])
    if not domains or not names:
        return False
    params = {n: models[n]["parameters"] for n in names}
    colors = bar_colors(names, params)
    ours = [n for n in names if n.startswith("Ours")]
    largest_ours = ours[-1] if ours else None
    medium = next((n for n in names if n.startswith("GPT-2 medium")), None)

    cols = 2
    rows = (len(domains) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.9 * rows + 0.6), facecolor=SURFACE,
                             layout="constrained", squeeze=False)
    y = list(range(len(names)))
    for k, ax in enumerate(axes.flat):
        if k >= len(domains):
            ax.set_visible(False)
            continue
        d = domains[k]
        values = [models[n][d]["bits_per_byte"] if d in models[n] else None for n in names]
        present = [v for v in values if v is not None]
        for i, (v, c) in enumerate(zip(values, colors)):
            if v is None:
                continue
            ax.hlines(i, min(present) - 0.02, v, color=GRID, linewidth=1)
            ax.plot(v, i, "o", color=c, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.2)
            ax.annotate(f"{v:.3f}", (v, i), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=TEXT)
        title = DOMAIN_TITLES[d]
        if largest_ours and medium and d in models[largest_ours] and d in models[medium]:
            diff = models[largest_ours][d]["bits_per_byte"] - models[medium][d]["bits_per_byte"]
            verdict = "better" if diff < 0 else "worse"
            title += f"\n{largest_ours.split(' (')[1][:-1]} vs GPT-2 medium: {diff:+.3f} ({verdict})"
        ax.set_yticks(y, names if k % cols == 0 else [""] * len(names))
        ax.invert_yaxis()
        style_axes(ax, title, "")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        span = max(present) - min(present)
        ax.set_xlim(min(present) - 0.02, max(present) + max(span * 0.25, 0.03))
        ax.set_xlabel("Bits per byte (lower is better)", fontsize=9, color=TEXT_SECONDARY)

    fig.suptitle("Bits per byte by kind of text, our models vs GPT-2",
                 x=0.01, ha="left", fontsize=13, color=TEXT, fontweight="bold")
    fig.get_layout_engine().set(h_pad=0.15, w_pad=0.2, hspace=0.1, wspace=0.08)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return True


# ------------------------------------------------------------
# 5. Zero-shot benchmarks through lm-evaluation-harness (harness_eval.py)
# ------------------------------------------------------------

BENCH_TITLES = {"hellaswag": "HellaSwag", "piqa": "PIQA", "arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge",
                "winogrande": "WinoGrande", "sciq": "SciQ", "openbookqa": "OpenBookQA",
                "lambada_openai": "LAMBADA (OpenAI)"}
BENCH_CHANCE = {"hellaswag": 25, "piqa": 50, "arc_easy": 25, "arc_challenge": 25, "winogrande": 50,
                "sciq": 25, "openbookqa": 25, "lambada_openai": 0}


def benchmark_comparison(harness, path):
    models = harness["models"]
    names = sorted((n for n in models if models[n].get("tasks")), key=lambda n: models[n]["parameters"])
    tasks = [t for t in BENCH_TITLES if any(t in models[n]["tasks"] for n in names)]
    if not names or not tasks:
        return False
    colors = bar_colors(names, {n: models[n]["parameters"] for n in names})
    cols = 4
    rows = (len(tasks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.9 * rows + 0.7), facecolor=SURFACE,
                             layout="constrained", squeeze=False)
    y = list(range(len(names)))
    for k, ax in enumerate(axes.flat):
        if k >= len(tasks):
            ax.set_visible(False)
            continue
        t = tasks[k]
        metric = "acc_norm" if any("acc_norm" in models[n]["tasks"].get(t, {}) for n in names) else "acc"
        vals = []
        for i, (n, c) in enumerate(zip(names, colors)):
            r = models[n]["tasks"].get(t)
            if not r or metric not in r:
                continue
            v, se = r[metric] * 100, r.get(metric + "_stderr", 0) * 100
            vals.append(v)
            ax.errorbar(v, i, xerr=se, fmt="o", color=c, ecolor=c, elinewidth=1.2, capsize=0, markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1)
            ax.annotate(f"{v:.1f}", (v + se, i), xytext=(5, 0), textcoords="offset points", va="center",
                        fontsize=8, color=TEXT)
        chance = BENCH_CHANCE.get(t, 0)
        lo = min(vals + [chance]) - 3 if chance else min(vals) - 5
        ax.set_xlim(max(lo, 0), max(vals) + (max(vals) - lo) * 0.3)
        if chance:
            ax.axvline(chance, color=TEXT_MUTED, linewidth=1, linestyle=(0, (4, 3)))
            # Just above the plot area, so it never sits on a data label
            ax.annotate(f"chance {chance}%", (chance, 1.0), xycoords=("data", "axes fraction"), xytext=(3, 1),
                        textcoords="offset points", fontsize=7.5, color=TEXT_SECONDARY, va="bottom")
        ax.set_yticks(y, names if k % cols == 0 else [""] * len(names))
        ax.invert_yaxis()
        style_axes(ax, f"{BENCH_TITLES[t]} ({metric})", "")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_xlabel("Accuracy (%)", fontsize=8.5, color=TEXT_SECONDARY)

    fig.suptitle("Zero-shot benchmarks (lm-evaluation-harness), our models vs GPT-2; bars show ±1 standard error",
                 x=0.01, ha="left", fontsize=13, color=TEXT, fontweight="bold")
    fig.get_layout_engine().set(h_pad=0.15, w_pad=0.15, hspace=0.1, wspace=0.05)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append",
                   help="LABEL=path/to/log.jsonl, repeatable (default: the 110M and 336M logs in logs/)")
    p.add_argument("--eval", default="results/eval_results.json")
    p.add_argument("--context", default="results/context_results.json")
    p.add_argument("--domains", default="results/domain_results.json")
    p.add_argument("--harness", default="results/harness_results.json")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--batch-tokens", type=int, default=524_288)
    args = p.parse_args()

    # Inter if it's installed, otherwise matplotlib's built-in DejaVu Sans (no "findfont" warnings)
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = ["Inter"] if "Inter" in installed else ["DejaVu Sans"]
    plt.rcParams["font.weight"] = "normal"
    os.makedirs(args.out_dir, exist_ok=True)

    run_args = args.run or ["110M=logs/log_110m.jsonl", "336M=logs/log_336m.jsonl"]
    runs = []
    for item in run_args:
        label, path = item.split("=", 1)
        if os.path.exists(path):
            runs.append((label, [json.loads(line) for line in open(path)]))
        else:
            print(f"{path} not found: skipped run {label}")
    if not runs:
        raise SystemExit("No training logs found.")

    results = json.load(open(args.eval)) if os.path.exists(args.eval) else None
    gpt2, params = {}, {}
    if results:
        for name, r in results["models"].items():
            params[name] = r.get("parameters", 0)
            if name.startswith("GPT-2") and "bpb" in r:
                gpt2[name.split(" (")[0]] = r["bpb"]["bits_per_byte"]

    curves = os.path.join(args.out_dir, "training_curves.png")
    training_curves(runs, args.batch_tokens, gpt2, curves)
    print(f"Wrote {curves}")

    if results:
        comparison = os.path.join(args.out_dir, "model_comparison.png")
        model_comparison(results, comparison)
        print(f"Wrote {comparison}")
    else:
        print(f"{args.eval} not found: skipped the model comparison")

    if os.path.exists(args.context):
        context = json.load(open(args.context))
        out = os.path.join(args.out_dir, "context_comparison.png")
        if context_comparison(context, params, out):
            print(f"Wrote {out}")
    else:
        print(f"{args.context} not found: skipped the context comparison")

    if os.path.exists(args.domains):
        out = os.path.join(args.out_dir, "domain_comparison.png")
        if domain_comparison(json.load(open(args.domains)), out):
            print(f"Wrote {out}")
    else:
        print(f"{args.domains} not found: skipped the domain comparison")

    if os.path.exists(args.harness):
        out = os.path.join(args.out_dir, "benchmark_comparison.png")
        if benchmark_comparison(json.load(open(args.harness)), out):
            print(f"Wrote {out}")
    else:
        print(f"{args.harness} not found: skipped the benchmark comparison")


if __name__ == "__main__":
    main()
