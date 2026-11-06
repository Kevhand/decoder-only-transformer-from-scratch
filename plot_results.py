"""
Plot the training run and the comparison with GPT-2, as PNGs for the write-up.

    pip install matplotlib
    python plot_results.py

Reads checkpoints/log.jsonl (training log) and results/eval_results.json (from
evaluate.py, optional). Writes:
    results/training_curves.png   loss, validation bits/byte, learning rate, gradient norm vs tokens
    results/model_comparison.png  bits/byte and HellaSwag for our model and GPT-2 small/medium/large
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
SERIES_1 = "#2a78d6"   # our model / training loss
SERIES_2 = "#eb6834"   # validation
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
    return f"{x:.1f}B" if x else "0"


def smooth(values, window):
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def label_end(ax, x, y, text, color, below=False):
    # Right-aligned at the last point, so it never runs past the plot area.
    ax.annotate(text, (x, y), xytext=(0, -9 if below else 9), textcoords="offset points", fontsize=8.5,
                color=color, va="top" if below else "bottom", ha="right", fontweight="bold")


def training_curves(records, batch_tokens, gpt2, path):
    train = [r for r in records if "train_loss" in r]
    val = [r for r in records if "val_loss" in r]
    tok = lambda rs: [r["step"] * batch_tokens / 1e9 for r in rs]  # noqa: E731

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE, layout="constrained")
    (a, b), (c, d) = axes

    # 1. Training and validation loss
    t_x, t_y = tok(train), [r["train_loss"] for r in train]
    t_s = smooth(t_y, 10)
    a.plot(t_x, t_y, color=SERIES_1, alpha=0.18, linewidth=1)
    a.plot(t_x, t_s, color=SERIES_1, linewidth=2, label="Training (smoothed)")
    v_x, v_y = tok(val), [r["val_loss"] for r in val]
    a.plot(v_x, v_y, color=SERIES_2, linewidth=2, marker="o", markersize=4,
           markeredgecolor=SURFACE, markeredgewidth=1.5, label="Validation")
    a.set_ylim(min(v_y + t_s[len(t_s) // 10:]) - 0.35, 7)  # start the axis after the first drop
    style_axes(a, "Loss", "Cross-entropy (nats/token)")
    a.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper right")
    label_end(a, v_x[-1], v_y[-1], f"Validation {v_y[-1]:.3f}", TEXT, below=True)

    # 2. Validation bits per byte, with GPT-2 for reference
    bpb = [r["val_bpb"] for r in val]
    b.plot(v_x, bpb, color=SERIES_1, linewidth=2, marker="o", markersize=4,
           markeredgecolor=SURFACE, markeredgewidth=1.5)
    label_end(b, v_x[-1], bpb[-1], f"Ours {bpb[-1]:.3f}", TEXT)
    for name, value in gpt2.items():
        b.axhline(value, color=REFERENCE, linewidth=1, linestyle=(0, (4, 3)))
        b.annotate(f"{name} {value:.3f}", (0.02, value), xycoords=("axes fraction", "data"),
                   xytext=(0, 4), textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
    b.set_ylim(min(bpb + list(gpt2.values())) - 0.05, 1.6)
    style_axes(b, "Validation bits per byte", "Bits per byte (lower is better)")

    # 3. Learning rate
    c.plot(t_x, [r["lr"] for r in train], color=SERIES_1, linewidth=2)
    style_axes(c, "Learning rate", "Learning rate")
    c.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}" if v else "0"))

    # 4. Gradient norm
    g = [r["grad_norm"] for r in train]
    c_g = smooth(g, 10)
    d.plot(t_x, g, color=SERIES_1, alpha=0.18, linewidth=1)
    d.plot(t_x, c_g, color=SERIES_1, linewidth=2)
    d.set_ylim(0, max(c_g[len(c_g) // 20:]) * 1.6)
    style_axes(d, "Gradient norm (before clipping at 1.0)", "L2 norm")

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(FuncFormatter(billions))
        ax.set_xlabel("Training tokens", fontsize=9, color=TEXT_SECONDARY)

    fig.suptitle("Training a 110M-parameter decoder from scratch on 2.5B tokens of FineWeb-Edu",
                 x=0.01, ha="left", fontsize=13, color=TEXT, fontweight="bold")
    fig.get_layout_engine().set(h_pad=0.15, w_pad=0.2, hspace=0.08, wspace=0.06)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def model_comparison(results, path):
    rows = []
    for name, r in results["models"].items():
        if "bpb" in r and "hellaswag" in r:
            rows.append((name, r["bpb"]["bits_per_byte"], r["hellaswag"]["acc_norm"] * 100,
                         name.startswith("Ours")))
    rows.sort(key=lambda row: results["models"][row[0]]["parameters"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE, layout="constrained")
    names = [r[0] for r in rows]
    colors = [SERIES_1 if r[3] else REFERENCE for r in rows]
    y = range(len(rows))

    for ax, values, title, xlabel, fmt in (
            (a, [r[1] for r in rows], "Bits per byte on held-out FineWeb-Edu", "Lower is better", "{:.3f}"),
            (b, [r[2] for r in rows], "HellaSwag accuracy (acc_norm)", "Higher is better", "{:.1f}%")):
        ax.barh(list(y), values, color=colors, height=0.55)
        for i, v in enumerate(values):
            ax.annotate(fmt.format(v), (v, i), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT)
        ax.set_yticks(list(y), names)
        ax.invert_yaxis()
        style_axes(ax, title, "")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)

    a.set_xlim(0, max(r[1] for r in rows) * 1.18)
    b.set_xlim(0, max(r[2] for r in rows) * 1.18)
    b.axvline(25, color=TEXT_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    b.annotate("random 25%", (25, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -4),
               textcoords="offset points", fontsize=8, color=TEXT_SECONDARY, va="top", annotation_clip=False)
    b.set_yticklabels([])

    fig.suptitle("Our model vs GPT-2, evaluated on identical data",
                 x=0.01, ha="left", fontsize=13, color=TEXT, fontweight="bold")
    fig.get_layout_engine().set(w_pad=0.2, wspace=0.08)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="checkpoints/log.jsonl")
    p.add_argument("--eval", default="results/eval_results.json")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--batch-tokens", type=int, default=524_288)
    args = p.parse_args()

    plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
    plt.rcParams["font.weight"] = "normal"
    os.makedirs(args.out_dir, exist_ok=True)
    records = [json.loads(line) for line in open(args.log)]

    results = json.load(open(args.eval)) if os.path.exists(args.eval) else None
    gpt2 = {}
    if results:
        for name, r in results["models"].items():
            if name.startswith("GPT-2") and "bpb" in r:
                gpt2[name.split(" (")[0]] = r["bpb"]["bits_per_byte"]

    curves = os.path.join(args.out_dir, "training_curves.png")
    training_curves(records, args.batch_tokens, gpt2, curves)
    print(f"Wrote {curves}")

    if results:
        comparison = os.path.join(args.out_dir, "model_comparison.png")
        model_comparison(results, comparison)
        print(f"Wrote {comparison}")
    else:
        print(f"{args.eval} not found: skipped the model comparison")


if __name__ == "__main__":
    main()
