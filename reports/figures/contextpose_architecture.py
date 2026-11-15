"""ContextPose architecture diagram for the methodology section.

Run from the repository root (matplotlib is not a project dependency):

    uv run --no-project --with matplotlib python reports/figures/contextpose_architecture.py

Writes contextpose_architecture.{pdf,svg,png} next to this script.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).with_suffix("")
# Arial's font header has near-zero timestamps, which fontTools reports
# while subsetting the font into the PDF. The output is unaffected.
logging.getLogger("fontTools").setLevel(logging.ERROR)

plt.rcParams.update(
    {
        "font.family": "Arial",
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
    }
)

# Palette: one hue per stream, a neutral grey for frozen extractors.
INK = "#222222"
MUTED = "#5f5f5f"
ARROW = "#3d3d3d"
STYLE = {
    "beh": ("#E4EDF8", "#3A6EA8"),
    "ctx": ("#FCEBD6", "#C2741F"),
    "face": ("#E2F1E6", "#3B8A57"),
    "fuse": ("#EDE7F5", "#6A52A0"),
    "frozen": ("#EFEFEF", "#8C8C8C"),
    "out": ("#FFFFFF", "#2B2B2B"),
    "fixed": ("#FFFFFF", "#C2741F"),
}

W_MM, H_MM = 171.0, 132.0
fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_MM)
ax.set_ylim(0, H_MM)
ax.set_aspect("equal")
ax.axis("off")

FS = 7.0  # body text, pt
FS_S = 6.0  # secondary text, pt


def box(cx, cy, w, h, kind, ls="-", lw=0.9, r=1.4, z=2):
    fc, ec = STYLE[kind]
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        fc=fc,
        ec=ec,
        lw=lw,
        ls=ls,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def text(x, y, s, size=FS, color=INK, weight="normal", ha="center", va="center", **kw):
    ax.text(
        x, y, s, size=size, color=color, weight=weight, ha=ha, va=va, zorder=6, **kw
    )


def label(cx, cy, title, sub=None, gap=2.1, title_size=FS, sub_size=FS_S):
    """Bold title with an optional grey subtitle, centred on (cx, cy)."""
    if sub is None:
        text(cx, cy, title, size=title_size, weight="bold")
    else:
        text(cx, cy + gap / 2 + 0.2, title, size=title_size, weight="bold")
        text(cx, cy - gap / 2 - 0.2, sub, size=sub_size, color=MUTED)


def arrow(p, q, color=ARROW, lw=0.9, ls="-", z=4):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>,head_length=2.2,head_width=1.1",
            mutation_scale=1,
            color=color,
            lw=lw,
            ls=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
    )


def path_arrow(points, color=ARROW, lw=0.9, ls="-"):
    """Polyline whose last segment ends in an arrow head."""
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=4)
    arrow(points[-2], points[-1], color=color, lw=lw, ls=ls)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
COLS = (62.0, 90.0, 118.0)  # pre-event, event, post-event
REGIONS = ("pre", "event", "post")
SPAN_L, SPAN_R = 47.0, 133.0  # wide boxes over the three windows
SPAN_C, SPAN_W = (SPAN_L + SPAN_R) / 2, SPAN_R - SPAN_L
CTX_X = 22.0
FACE_X = 152.0

Y_TIME = 124.0
Y_EXTRACT = 111.0
Y_X = 97.0
Y_ENC = 84.0
Y_ATT = 66.0
Y_REG = 47.5
Y_HEAD = 36.0
Y_SUM = 26.0
Y_OUT = 14.5

# --------------------------------------------------------------------------
# Clip timeline
# --------------------------------------------------------------------------
seg_fc = ("#D7DCE3", "#B9C3D0", "#D7DCE3")
seg_txt = (
    r"pre-event  $\cdot$  5 s",
    "event  (variable length)",
    r"post-event  $\cdot$  5 s",
)
edges = (SPAN_L, 76.0, 104.0, SPAN_R)
for i in range(3):
    ax.add_patch(
        Rectangle(
            (edges[i], Y_TIME - 3),
            edges[i + 1] - edges[i],
            6,
            fc=seg_fc[i],
            ec="white",
            lw=1.2,
            zorder=2,
        )
    )
    text((edges[i] + edges[i + 1]) / 2, Y_TIME, seg_txt[i], size=FS_S + 0.3)
ax.add_patch(
    FancyBboxPatch(
        (SPAN_L, Y_TIME - 3),
        SPAN_W,
        6,
        boxstyle="round,pad=0,rounding_size=1.2",
        fc="none",
        ec="#8C96A3",
        lw=0.8,
        zorder=3,
    )
)
text(SPAN_L - 2, Y_TIME, "Video clip", weight="bold", ha="right")

# --------------------------------------------------------------------------
# Frozen per-step feature extraction (pose OR EmotionCLIP)
# --------------------------------------------------------------------------
box(SPAN_C, Y_EXTRACT, SPAN_W, 12, "frozen")
text(
    SPAN_L + 2.2,
    Y_EXTRACT + 4.0,
    "Per-step behaviour features",
    size=FS_S,
    color=MUTED,
    ha="left",
)
text(
    SPAN_R - 2.2,
    Y_EXTRACT + 4.0,
    r"$T = 32$ steps per window",
    size=FS_S,
    color=MUTED,
    ha="right",
)
pill_w, pill_h = 35.5, 6.4
for cx, title, sub in (
    (SPAN_C - 21.5, "Pose (11 joints)", r"$x, y$, confidence, speed  $\to D_x = 44$"),
    (
        SPAN_C + 21.5,
        "EmotionCLIP (subject mask)",
        r"frame embedding + flag  $\to D_x = 513$",
    ),
):
    box(cx, Y_EXTRACT - 1.2, pill_w, pill_h, "out", lw=0.7, r=1.1, z=3)
    text(cx, Y_EXTRACT - 1.2 + 1.45, title, size=FS_S + 0.3, weight="bold")
    text(cx, Y_EXTRACT - 1.2 - 1.55, sub, size=FS_S - 0.4, color=MUTED)
text(SPAN_C, Y_EXTRACT - 1.2, "or", size=FS, weight="bold", color=MUTED, style="italic")

for cx in COLS:
    arrow((cx, Y_TIME - 3), (cx, Y_EXTRACT + 6))

# --------------------------------------------------------------------------
# Behaviour sequences X_r
# --------------------------------------------------------------------------
N_CELLS = 16
for cx, reg in zip(COLS, REGIONS):
    arrow((cx, Y_EXTRACT - 6), (cx, Y_X + 4))
    box(cx, Y_X, 23, 8, "beh", lw=0.8)
    cw, ch = 1.05, 3.0
    x0 = cx - N_CELLS * cw / 2
    for k in range(N_CELLS):
        ax.add_patch(
            Rectangle(
                (x0 + k * cw + 0.12, Y_X - 3.1),
                cw - 0.24,
                ch,
                fc="#9DB8DA",
                ec="none",
                zorder=3,
            )
        )
    text(cx, Y_X + 1.8, rf"$\mathbf{{X}}_{{\mathrm{{{reg}}}}}$", size=FS + 0.5)

# --------------------------------------------------------------------------
# Shared temporal encoder
# --------------------------------------------------------------------------
box(SPAN_C, Y_ENC, SPAN_W, 7.5, "beh")
label(
    SPAN_C,
    Y_ENC,
    "Temporal encoder  (shared across windows)",
    r"2 $\times$ [Conv1d, kernel 3, GELU]   $\to$   tokens $\mathbf{Z}_r \in \mathbb{R}^{T \times W}$",
)
for cx in COLS:
    arrow((cx, Y_X - 4), (cx, Y_ENC + 3.75))

# --------------------------------------------------------------------------
# Context-queried cross-attention (with a sketch of the attention weights)
# --------------------------------------------------------------------------
ATT_H = 19.0
box(SPAN_C, Y_ATT, SPAN_W, ATT_H, "fuse")
text(
    SPAN_C,
    Y_ATT + ATT_H / 2 - 2.6,
    "Context-queried temporal cross-attention",
    weight="bold",
)
text(
    SPAN_C,
    Y_ATT + ATT_H / 2 - 5.2,
    "4 heads, shared across windows;  context sets the weights over time steps",
    size=FS_S,
    color=MUTED,
)

# Illustrative attention profiles: different events emphasise different steps.
import math

profiles = (
    lambda t: 0.25 + 0.15 * math.sin(6.0 * t),
    lambda t: 0.15 + 0.85 * math.exp(-((t - 0.28) ** 2) / 0.012),
    lambda t: 0.12
    + 0.7 * math.exp(-((t - 0.72) ** 2) / 0.02)
    + 0.25 * math.exp(-((t - 0.2) ** 2) / 0.01),
)
for cx, prof in zip(COLS, profiles):
    cw = 1.05
    x0 = cx - N_CELLS * cw / 2
    base = Y_ATT - 6.3
    for k in range(N_CELLS):
        a = prof(k / (N_CELLS - 1))
        ax.add_patch(
            Rectangle(
                (x0 + k * cw + 0.12, base + 2.4),
                cw - 0.24,
                5.0 * a,
                fc="#B88AD9",
                ec="none",
                zorder=3,
            )
        )
        ax.add_patch(
            Rectangle(
                (x0 + k * cw + 0.12, base),
                cw - 0.24,
                1.8,
                fc="#9DB8DA",
                ec="none",
                zorder=3,
            )
        )
    text(x0 - 1.0, base + 4.6, r"$\alpha$", size=FS, ha="right", color="#6A52A0")
    text(
        x0 - 1.0,
        base + 0.9,
        r"$\mathbf{Z}$",
        size=FS_S + 0.5,
        ha="right",
        color="#3A6EA8",
    )

for cx in COLS:
    arrow((cx, Y_ENC - 3.75), (cx, Y_ATT + ATT_H / 2))
    text(
        cx + 1.2,
        (Y_ENC - 3.75 + Y_ATT + ATT_H / 2) / 2,
        r"$\mathbf{Z}_r$",
        size=FS_S + 0.5,
        ha="left",
    )

# --------------------------------------------------------------------------
# Context stream (left)
# --------------------------------------------------------------------------
box(CTX_X, Y_EXTRACT, 34, 12, "ctx")
text(CTX_X, Y_EXTRACT + 3.0, r"Event context  $\mathbf{c}$", weight="bold")
text(CTX_X, Y_EXTRACT - 0.6, r"indicators ($d_c = 4$)  or", size=FS_S, color=MUTED)
text(CTX_X, Y_EXTRACT - 3.5, r"VLM description ($d_c = 384$)", size=FS_S, color=MUTED)

LIN_X = 27.0
box(LIN_X, Y_ENC, 22, 7.5, "ctx")
label(
    LIN_X,
    Y_ENC,
    "Linear",
    r"$\hat{\mathbf{c}} = \mathbf{W}_c\mathbf{c} + \mathbf{b}_c$",
)
arrow((LIN_X, Y_EXTRACT - 6), (LIN_X, Y_ENC + 3.75))

# Query = projected context + region embedding.
PLUS_Q = (LIN_X, Y_ATT)
ax.add_patch(Circle(PLUS_Q, 2.2, fc="white", ec=ARROW, lw=0.9, zorder=5))
text(*PLUS_Q, "+", size=FS + 1.5, weight="bold")
arrow((LIN_X, Y_ENC - 3.75), (LIN_X, Y_ATT + 2.2))
text(
    LIN_X + 1.2,
    (Y_ENC - 3.75 + Y_ATT + 2.2) / 2,
    r"$\hat{\mathbf{c}}$",
    size=FS + 0.5,
    ha="left",
)

EMB_X = 16.5
box(EMB_X, Y_ATT, 11.5, 7.5, "ctx")
label(EMB_X, Y_ATT, "Region", r"emb. $\mathbf{e}_r$", gap=2.0)
arrow((EMB_X + 5.75, Y_ATT), (LIN_X - 2.2, Y_ATT))

arrow((LIN_X + 2.2, Y_ATT), (SPAN_L, Y_ATT))
text((LIN_X + 2.2 + SPAN_L) / 2, Y_ATT + 2.0, "query", size=FS_S, color=MUTED)
text(
    (LIN_X + 2.2 + SPAN_L) / 2,
    Y_ATT - 2.2,
    r"$\mathbf{q}_r = \hat{\mathbf{c}} + \mathbf{e}_r$",
    size=FS,
)

# Fixed lookup offset from the context.
LOOK_X, LOOK_W = 26.0, 28.0
box(LOOK_X, Y_SUM, LOOK_W, 10, "fixed")
text(LOOK_X, Y_SUM + 2.6, "Context lookup", weight="bold")
text(
    LOOK_X,
    Y_SUM - 0.6,
    r"offset $\mathbf{b}_{g(\mathbf{c})}$, not trained;",
    size=FS_S,
    color=MUTED,
)
text(LOOK_X, Y_SUM - 3.0, "fit on training folds only", size=FS_S, color=MUTED)
LOOK_LINE_X = 6.8
path_arrow(
    [
        (LOOK_LINE_X, Y_EXTRACT - 6),
        (LOOK_LINE_X, Y_SUM),
        (LOOK_X - LOOK_W / 2, Y_SUM),
    ],
    color=STYLE["ctx"][1],
)

# --------------------------------------------------------------------------
# Inter-region Transformer, head, sum, output
# --------------------------------------------------------------------------
box(SPAN_C, Y_REG, SPAN_W, 7.5, "fuse")
label(
    SPAN_C,
    Y_REG,
    "Inter-region Transformer  (1 layer, 4 heads)",
    r"relates $[\mathbf{s}_{\mathrm{pre}}; \mathbf{s}_{\mathrm{event}}; \mathbf{s}_{\mathrm{post}}]$, then mean over windows $\to \mathbf{h}$",
)
for cx in COLS:
    arrow((cx, Y_ATT - ATT_H / 2), (cx, Y_REG + 3.75))
    text(
        cx + 1.2,
        (Y_ATT - ATT_H / 2 + Y_REG + 3.75) / 2,
        r"$\mathbf{s}_r$",
        size=FS_S + 0.5,
        ha="left",
    )

box(SPAN_C, Y_HEAD, 30, 6.5, "fuse")
label(SPAN_C, Y_HEAD, "MLP head", r"$W \to W/2 \to 2$", gap=2.0)
arrow((SPAN_C, Y_REG - 3.75), (SPAN_C, Y_HEAD + 3.25))
text(
    SPAN_C + 1.2,
    (Y_REG - 3.75 + Y_HEAD + 3.25) / 2,
    r"$\mathbf{h}$",
    size=FS_S + 0.5,
    ha="left",
)

R_SUM = 2.6
ax.add_patch(Circle((SPAN_C, Y_SUM), R_SUM, fc="white", ec=ARROW, lw=0.9, zorder=5))
text(SPAN_C, Y_SUM, "+", size=FS + 2, weight="bold")
arrow((SPAN_C, Y_HEAD - 3.25), (SPAN_C, Y_SUM + R_SUM))
text(
    SPAN_C + 1.2,
    (Y_HEAD - 3.25 + Y_SUM + R_SUM) / 2,
    r"$\hat{\mathbf{y}}_{\mathrm{beh}}$",
    size=FS_S + 0.5,
    ha="left",
)

arrow((LOOK_X + LOOK_W / 2, Y_SUM), (SPAN_C - R_SUM, Y_SUM), color=STYLE["ctx"][1])

box(SPAN_C, Y_OUT, 40, 8.5, "out", lw=1.1)
text(
    SPAN_C,
    Y_OUT + 1.7,
    r"$\hat{\mathbf{y}} = [\hat{y}_{\mathrm{val}},\ \hat{y}_{\mathrm{aro}}]$",
    size=FS + 1,
)
text(
    SPAN_C, Y_OUT - 2.1, "logits of the annotators' vote share", size=FS_S, color=MUTED
)
arrow((SPAN_C, Y_SUM - R_SUM), (SPAN_C, Y_OUT + 4.25))

# --------------------------------------------------------------------------
# Optional gated face stream (right)
# --------------------------------------------------------------------------
FW = 28.0
group = FancyBboxPatch(
    (FACE_X - FW / 2 - 2.5, Y_REG - 5.6),
    FW + 5,
    (Y_EXTRACT + 9.3) - (Y_REG - 5.6),
    boxstyle="round,pad=0,rounding_size=2.0",
    fc="none",
    ec=STYLE["face"][1],
    lw=0.9,
    ls=(0, (3, 2)),
    zorder=1,
)
ax.add_patch(group)
text(
    FACE_X,
    Y_EXTRACT + 11.2,
    "Face stream",
    weight="bold",
    color=STYLE["face"][1],
)

box(FACE_X, Y_EXTRACT, FW, 12, "frozen")
text(FACE_X, Y_EXTRACT + 3.2, "EQ-Former", weight="bold")
text(FACE_X, Y_EXTRACT - 0.4, "face crop at each step", size=FS_S, color=MUTED)
text(FACE_X, Y_EXTRACT - 3.3, r"quality gate $m_{r,t}$", size=FS_S, color=MUTED)

box(FACE_X, Y_ENC, FW, 7.5, "face")
label(FACE_X, Y_ENC, "Step projection", r"LN $\to$ Linear $\to$ GELU")
arrow((FACE_X, Y_EXTRACT - 6), (FACE_X, Y_ENC + 3.75))
text(
    FACE_X + 1.2,
    (Y_EXTRACT - 6 + Y_ENC + 3.75) / 2,
    r"$\mathbf{x}^{f}_{r,t}$",
    size=FS_S + 0.5,
    ha="left",
)

box(FACE_X, Y_ATT, FW, 7.5, "face")
label(FACE_X, Y_ATT, "Gated mean", r"per window $\to \bar{\mathbf{u}}_r$")
arrow((FACE_X, Y_ENC - 3.75), (FACE_X, Y_ATT + 3.75))
text(
    FACE_X + 1.2,
    (Y_ENC - 3.75 + Y_ATT + 3.75) / 2,
    r"$\mathbf{u}_{r,t} \in \mathbb{R}^{32}$",
    size=FS_S + 0.5,
    ha="left",
)

box(FACE_X, Y_REG, FW, 7.5, "face")
label(FACE_X, Y_REG, r"Face head $\psi$", "zero-initialised")
arrow((FACE_X, Y_ATT - 3.75), (FACE_X, Y_REG + 3.75))

path_arrow(
    [
        (FACE_X, Y_REG - 3.75),
        (FACE_X, Y_SUM),
        (SPAN_C + R_SUM, Y_SUM),
    ],
    color=STYLE["face"][1],
)
text(
    FACE_X - 2.0,
    Y_SUM + 2.0,
    r"$\mathbf{\Delta}_{\mathrm{face}}$ (zero if no valid face)",
    size=FS_S,
    ha="right",
    color=STYLE["face"][1],
)

# --------------------------------------------------------------------------
# Legend
# --------------------------------------------------------------------------
LX, LY = 118.0, 12.0
items = (
    ("frozen", "-", "frozen, pretrained"),
    ("beh", "-", "trainable"),
    ("fixed", "-", "fixed, not trained"),
)
for i, (kind, ls, name) in enumerate(items):
    y = LY + 5.0 - i * 4.6
    box(LX, y, 5.5, 3.2, kind, ls=ls, lw=0.8, r=0.7)
    text(LX + 4.2, y, name, size=FS_S, ha="left", color=MUTED)
ax.plot(
    [LX + 22, LX + 27.5],
    [LY + 5.0, LY + 5.0],
    color=STYLE["face"][1],
    lw=0.9,
    ls=(0, (3, 2)),
)
text(LX + 29.0, LY + 5.0, "optional", size=FS_S, ha="left", color=MUTED)

for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
    fig.savefig(OUT.with_suffix(f".{ext}"), **kw)
print("wrote", ", ".join(OUT.with_suffix(f".{e}").name for e in ("pdf", "svg", "png")))
