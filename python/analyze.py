"""Tables and figures from the dot-product sweep and the training sweep.

Reads results/dot_sweep.csv and results/runs/*.jsonl, writes
results/dot_table.md, results/train_table.md and the figures in docs/img.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
IMG = ROOT / "docs" / "img"

# Reference palette, light mode, validated with the dataviz validator: slots in order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
STYLES = ["-", "--", "-.", ":"]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ dot sweep


def dot_sweep():
    rows = list(csv.DictReader((RESULTS / "dot_sweep.csv").open()))
    med = defaultdict(list)
    clamp = defaultdict(list)
    for r in rows:
        med[(r["group"], r["label"], int(r["n"]))].append(float(r["rel_error"]))
        clamp[(r["label"], int(r["n"]))].append(int(r["clamped"]))

    labels = []
    for r in rows:
        key = (r["group"], r["label"])
        if key not in labels:
            labels.append(key)

    lines = ["| group | policy | n=256 | n=1024 | n=4096 | clamped per pair, n=4096 |",
             "|---|---|---:|---:|---:|---:|"]
    for group, label in labels:
        cells = [f"{statistics.median(med[(group, label, n)]):.2e}" for n in (256, 1024, 4096)]
        c = statistics.mean(clamp[(label, 4096)])
        lines.append(f"| {group} | {label} | " + " | ".join(cells) + f" | {c:.1f} |")
    (RESULTS / "dot_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    # Figure: accumulation alone, error against length.
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ns = [256, 1024, 4096]
    ref = [statistics.median(med[("accumulation", "fp32 sequential", n)]) for n in ns]
    ax.plot(ns, ref, color=MUTED, linewidth=2, linestyle=":", marker="o", markersize=5)
    ax.annotate("fp32 sequential", (ns[-1], ref[-1]), xytext=(8, 0), textcoords="offset points",
                color=INK_2, fontsize=9, va="center")
    series = ["fp16 sequential", "fp16 pairwise", "bf16 sequential", "bf16 pairwise"]
    # fp16 sequential and bf16 pairwise end within a few percent of each other, so
    # their end labels are pushed apart rather than drawn on top of one another.
    nudge = {"fp16 sequential": 6, "bf16 pairwise": -6}
    for i, label in enumerate(series):
        ys = [statistics.median(med[("accumulation", label, n)]) for n in ns]
        ax.plot(ns, ys, color=SERIES[i], linewidth=2, linestyle=STYLES[i], marker="o",
                markersize=5, label=label)
        ax.annotate(label, (ns[-1], ys[-1]), xytext=(8, nudge.get(label, 0)),
                    textcoords="offset points", color=INK_2, fontsize=9, va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlim(200, 12000)
    ax.set_xlabel("terms in the dot product", color=INK_2)
    ax.set_ylabel("median relative error, 200 seeds", color=INK_2)
    ax.set_title("Accumulation alone: nothing cast, only the sum rounds", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="center left", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(IMG / "accumulation.png", facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ training


def load_runs():
    runs = defaultdict(dict)  # policy -> seed -> parsed
    for path in sorted((RESULTS / "runs").glob("*.jsonl")):
        cfg, measures, evals, summary, diverged_at = None, [], [], None, None
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A run still being written. It has no summary yet and is skipped below.
                continue
            kind = rec["kind"]
            if kind == "config":
                cfg = rec
            elif kind == "measure":
                measures.append(rec)
            elif kind == "eval":
                evals.append(rec)
            elif kind == "summary":
                summary = rec
            elif kind == "diverged":
                diverged_at = rec["step"]
        if cfg is None or summary is None:
            continue
        runs[cfg["policy"]][cfg["seed"]] = {
            "cfg": cfg, "measures": measures, "evals": evals,
            "summary": summary, "diverged_at": diverged_at,
        }
    return runs


def fmt_range(values):
    if not values:
        return "—"
    m = statistics.mean(values)
    if len(values) == 1:
        return f"{m:.3f}"
    return f"{m:.3f} ({min(values):.3f}–{max(values):.3f})"


def training(order):
    runs = load_runs()
    base = runs.get("fp32", {})

    lines = ["| policy | diverged | held-out loss, mean (range) | vs fp32, paired | grad cos | grad gain | grad rel. error |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for policy in order:
        if policy not in runs:
            continue
        seeds = runs[policy]
        div = sum(1 for r in seeds.values() if r["summary"]["diverged"])
        ok = {s: r for s, r in seeds.items() if not r["summary"]["diverged"]}
        vals = [r["summary"]["val_loss"] for r in ok.values()]
        deltas = [r["summary"]["val_loss"] - base[s]["summary"]["val_loss"]
                  for s, r in ok.items() if s in base]
        # Gradient metrics over every run, diverged or not, up to where it stopped.
        cos = statistics.mean(r["summary"]["mean_cos"] for r in seeds.values())
        gain = statistics.mean(r["summary"]["mean_gain"] for r in seeds.values())
        rel = statistics.mean(r["summary"]["mean_rel"] for r in seeds.values())
        delta = f"{statistics.mean(deltas):+.3f}" if deltas else "—"
        lines.append(f"| `{policy}` | {div}/{len(seeds)} | {fmt_range(vals)} | {delta} | "
                     f"{cos:.4f} | {gain:.3f} | {rel:.3f} |")
    (RESULTS / "train_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    # Figure: gradient gain through training, averaged over seeds.
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.axhline(1.0, color=AXIS, linewidth=1)
    picks = ["fp8", "fp8-mx", "mxfp4", "mxfp4-sr"]
    for i, policy in enumerate(picks):
        if policy not in runs:
            continue
        by_step = defaultdict(list)
        for r in runs[policy].values():
            for m in r["measures"]:
                if m["gain"] is not None:
                    by_step[m["step"]].append(m["gain"])
        steps = sorted(by_step)
        ys = [statistics.mean(by_step[s]) for s in steps]
        # A short trailing mean so the line reads as a trend, not as batch noise.
        k = 8
        smooth = [statistics.mean(ys[max(0, j - k + 1): j + 1]) for j in range(len(ys))]
        ax.plot(steps, smooth, color=SERIES[i], linewidth=2, linestyle=STYLES[i], label=policy)
        ax.annotate(policy, (steps[-1], smooth[-1]), xytext=(8, 0), textcoords="offset points",
                    color=INK_2, fontsize=9, va="center")
    ax.set_xlim(0, 2350)
    ax.set_xlabel("training step", color=INK_2)
    ax.set_ylabel("gradient gain along the exact gradient", color=INK_2)
    ax.set_title("How much of the true gradient step survives the arithmetic", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(IMG / "gain.png", facecolor=SURFACE)
    plt.close(fig)

    # Per-layer gain for the policies that shrink, to see where the shrinkage enters.
    layer_lines = ["| policy | w1 gain | w2 gain | w3 gain |", "|---|---:|---:|---:|"]
    for policy in ["fp8", "fp8-mx", "fp8-mx-headroom", "mxfp4", "mxfp4-headroom", "mxfp4-sr"]:
        if policy not in runs:
            continue
        acc = defaultdict(list)
        for r in runs[policy].values():
            for m in r["measures"]:
                for layer, v in m["layers"].items():
                    if v["gain"] is not None:
                        acc[layer].append(v["gain"])
        layer_lines.append(f"| `{policy}` | " + " | ".join(
            f"{statistics.mean(acc[l]):.3f}" for l in ("w1", "w2", "w3")) + " |")
    (RESULTS / "layer_table.md").write_text("\n".join(layer_lines) + "\n", encoding="utf-8", newline="\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    IMG.mkdir(parents=True, exist_ok=True)
    dot_sweep()
    order = [
        "exact", "fp32", "bf16", "fp8", "fp8-delayed", "fp8-e4m3-grads", "fp8-mx", "fp8-mx-headroom",
        "fp8-reversed", "fp8-acc-fp16", "fp8-acc-fp16-pairwise", "fp8-acc-bf16",
        "fp8-acc-bf16-pairwise", "fp8-acc-bf16-blocked", "fp8-acc-bf16-sr",
        "mxfp4", "mxfp4-headroom", "mxfp4-sr",
    ]
    if (RESULTS / "runs").exists():
        training(order)
    for name in ("dot_table.md", "train_table.md", "layer_table.md"):
        p = RESULTS / name
        if p.exists():
            print(f"== {name}\n{p.read_text(encoding="utf-8")}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
