"""Quadrant recall per model (no reference bars).

Values are the per-quadrant recalls of Table tab:quadrants in reports/test.tex.
Writes quadrant_recall_models.{pdf,png} next to this script.
"""
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).with_name("quadrant_recall_models")

# model -> (pos/high, pos/low, notpos/high, notpos/low)
MODELS = {
    "Annotation + face (V3)": (64, 29, 20, 57),
    "Annotation + VLM + face": (50, 36, 20, 62),
    "VLM + face": (57, 22, 20, 62),
    "Annotation + VLM": (25, 61, 20, 50),
    "Pose + annotation + face (full)": (29, 26, 0, 55),
    "Pose only": (14, 28, 60, 25),
}
N = {"pos_high": 28, "pos_low": 72, "neg_high": 5, "neg_low": 40}

# (row, col, title, index into the value tuple)
PANELS = [
    (0, 0, f"Not positive, high arousal  (n = {N['neg_high']})", 2),
    (0, 1, f"Positive valence, high arousal  (n = {N['pos_high']})", 0),
    (1, 0, f"Not positive, low arousal  (n = {N['neg_low']})", 3),
    (1, 1, f"Positive valence, low arousal  (n = {N['pos_low']})", 1),
]

BLUE = "#2878d6"
GRID = "#dddddd"
AXIS = "#888888"
TEXT = "#444444"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharex=True)

names = list(MODELS)
ys = range(len(names))
for r, c, title, k in PANELS:
    ax = axes[r][c]
    vals = [MODELS[m][k] for m in names]
    ax.barh(ys, vals, height=0.62, color=BLUE, zorder=3)
    for y, v in zip(ys, vals):
        ax.text(v + 1.8, y, f"{v}", va="center", fontsize=10, color=TEXT)
    ax.axvline(25, ls="--", color="#bbbbbb", lw=1.2, zorder=2)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(names if c == 0 else [], fontsize=11.5, color=TEXT)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=13)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    if r == 1:
        ax.set_xlabel("Recall within the quadrant (%)", color=TEXT, fontsize=11.5)

fig.suptitle("Quadrant recall of each model's combined valence and arousal calls",
             x=0.02, ha="left", fontsize=14, y=0.985)
fig.text(0.02, 0.935, "Dashed line: chance (25%).", fontsize=10.5, color=TEXT)
fig.text(0.008, 0.5, "arousal: low  ←  →  high", rotation=90,
         va="center", fontsize=10.5, color=TEXT)
fig.text(0.6, 0.012, "valence: not positive  ←  →  positive",
         ha="center", fontsize=10.5, color=TEXT)
fig.tight_layout(rect=(0.02, 0.03, 1, 0.92), h_pad=2.2, w_pad=1.5)

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=120)
print("wrote", OUT.with_suffix(".pdf"), OUT.with_suffix(".png"))
