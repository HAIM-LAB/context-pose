"""ContextPose architecture diagram for the methodology section.

Run from the repository root (matplotlib is not a project dependency):

    uv run --no-project --with matplotlib python reports/figures/contextpose_architecture.py

Writes contextpose_architecture.{pdf,svg,png} next to this script.
"""

import logging
import math
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
MUTED = "#4a4a4a"
ARROW = "#3d3d3d"
STYLE = {
    "beh": ("#E4EDF8", "#3A6EA8"),
    "ctx": ("#FCEBD6", "#C2741F"),
    "face": ("#E2F1E6", "#3B8A57"),
    "fuse": ("#EDE7F5", "#6A52A0"),
    "frozen": ("#EFEFEF", "#8C8C8C"),
    "out": ("#FFFFFF", "#2B2B2B"),
    "fixed": ("#FFFFFF", "#C2741F"),
    "data": ("#FFFFFF", "#3A6EA8"),
}

W_MM, H_MM = 171.0, 133.0
fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_MM)
ax.set_ylim(0, H_MM)
ax.set_aspect("equal")
ax.axis("off")

FS = 8.5  # titles, pt
FS_S = 7.5  # secondary text, pt
FS_M = 8.0  # math labels beside arrows, pt


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


def label(cx, cy, title, sub=None, gap=3.4, title_size=FS, sub_size=FS_S):
    """Bold title with an optional grey subtitle, centred on (cx, cy)."""
    if sub is None:
        text(cx, cy, title, size=title_size, weight="bold")
    else:
        text(cx, cy + gap / 2, title, size=title_size, weight="bold")
        text(cx, cy - gap / 2, sub, size=sub_size, color=MUTED)


def arrow(p, q, color=ARROW, lw=0.8, ls="-", z=4):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>,head_length=1.7,head_width=0.85",
            mutation_scale=1,
            color=color,
            lw=lw,
            ls=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
    )


def path_arrow(points, color=ARROW, lw=0.8, ls="-"):
    """Polyline whose last segment ends in an arrow head."""
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=4)
    arrow(points[-2], points[-1], color=color, lw=lw, ls=ls)


# --------------------------------------------------------------------------
# Geometry: rows are stacked from the top with a short arrow gap between them
# --------------------------------------------------------------------------
COLS = (62.0, 90.0, 118.0)  # pre-event, event, post-event
REGIONS = ("pre", "event", "post")
SPAN_L, SPAN_R = 47.0, 133.0  # wide boxes over the three windows
SPAN_C, SPAN_W = (SPAN_L + SPAN_R) / 2, SPAN_R - SPAN_L
CTX_X, CTX_W = 22.0, 40.0
FACE_X, FW = 153.5, 30.5

GAP = 3.4  # vertical arrow length between consecutive rows
H_TIME, H_EXTRACT, H_X, H_ENC, ATT_H = 7.0, 18.5, 9.0, 9.0, 23.0
H_REG, H_HEAD, R_SUM, H_OUT = 9.0, 8.0, 3.0, 11.0

Y_TIME = H_MM - 3.0 - H_TIME / 2
Y_EXTRACT = Y_TIME - H_TIME / 2 - GAP - H_EXTRACT / 2
Y_X = Y_EXTRACT - H_EXTRACT / 2 - GAP - H_X / 2
Y_ENC = Y_X - H_X / 2 - GAP - H_ENC / 2
Y_ATT = Y_ENC - H_ENC / 2 - GAP - ATT_H / 2
Y_REG = Y_ATT - ATT_H / 2 - GAP - H_REG / 2
Y_HEAD = Y_REG - H_REG / 2 - GAP - H_HEAD / 2
Y_SUM = Y_HEAD - H_HEAD / 2 - GAP - R_SUM
Y_OUT = Y_SUM - R_SUM - GAP - H_OUT / 2


def between(top, bottom):
    """Midpoint of the gap between the bottom of one row and the top of the next."""
    return (top + bottom) / 2


# --------------------------------------------------------------------------
# Clip timeline
# --------------------------------------------------------------------------
seg_fc = ("#D7DCE3", "#B9C3D0", "#D7DCE3")
seg_txt = (
    r"pre-event  $\cdot$  5 s",
    "event  (variable length)",
    r"post-event  $\cdot$  5 s",
)
edges = (SPAN_L, 73.0, 107.0, SPAN_R)
for i in range(3):
    ax.add_patch(
        Rectangle(
            (edges[i], Y_TIME - H_TIME / 2),
            edges[i + 1] - edges[i],
            H_TIME,
            fc=seg_fc[i],
            ec="white",
            lw=1.2,
            zorder=2,
        )
    )
    text((edges[i] + edges[i + 1]) / 2, Y_TIME, seg_txt[i], size=FS_S)
ax.add_patch(
    FancyBboxPatch(
        (SPAN_L, Y_TIME - H_TIME / 2),
        SPAN_W,
        H_TIME,
        boxstyle="round,pad=0,rounding_size=1.2",
        fc="none",
        ec="#8C96A3",
        lw=0.8,
        zorder=3,
    )
)
text(SPAN_L - 2, Y_TIME, "Video clip", weight="bold", ha="right")

# --------------------------------------------------------------------------
# Frozen per-step feature extraction (pose and/or EmotionCLIP)
# --------------------------------------------------------------------------
EX_TOP = Y_EXTRACT + H_EXTRACT / 2
box(SPAN_C, Y_EXTRACT, SPAN_W, H_EXTRACT, "frozen")
text(
    SPAN_L + 2.0,
    EX_TOP - 2.6,
    "Per-step behaviour features",
    size=FS_S,
    color=MUTED,
    ha="left",
)
text(
    SPAN_R - 2.0,
    EX_TOP - 2.6,
    r"$T = 32$ steps per window",
    size=FS_S,
    color=MUTED,
    ha="right",
)
pill_w, pill_h = 37.0, 11.5
pill_y = Y_EXTRACT - H_EXTRACT / 2 + 1.5 + pill_h / 2
for cx, title, line1, line2 in (
    (SPAN_C - 23.5, "Pose (11 joints)", r"$x, y$, confidence, speed", r"$D_x = 44$"),
    (
        SPAN_C + 23.5,
        "EmotionCLIP",
        "frame embedding + flag",
        r"subject mask,  $D_x = 513$",
    ),
):
    box(cx, pill_y, pill_w, pill_h, "out", lw=0.7, r=1.1, z=3)
    text(cx, pill_y + 3.4, title, size=FS, weight="bold")
    text(cx, pill_y - 0.2, line1, size=FS_S, color=MUTED)
    text(cx, pill_y - 3.6, line2, size=FS_S, color=MUTED)
text(
    SPAN_C,
    pill_y,
    "and/\nor",
    size=FS_S,
    weight="bold",
    color=MUTED,
    style="italic",
    linespacing=1.1,
)

for cx in COLS:
    arrow((cx, Y_TIME - H_TIME / 2), (cx, EX_TOP))

# --------------------------------------------------------------------------
# Behaviour sequences X_r
# --------------------------------------------------------------------------
N_CELLS = 16
for cx, reg in zip(COLS, REGIONS):
    arrow((cx, Y_EXTRACT - H_EXTRACT / 2), (cx, Y_X + H_X / 2))
    box(cx, Y_X, 23, H_X, "data", lw=0.8)
    cw, ch = 1.05, 3.0
    x0 = cx - N_CELLS * cw / 2
    for k in range(N_CELLS):
        ax.add_patch(
            Rectangle(
                (x0 + k * cw + 0.12, Y_X - 3.6),
                cw - 0.24,
                ch,
                fc="#9DB8DA",
                ec="none",
                zorder=3,
            )
        )
    text(cx, Y_X + 1.9, rf"$\mathbf{{X}}_{{\mathrm{{{reg}}}}}$", size=FS + 1)

# --------------------------------------------------------------------------
# Shared temporal encoder
# --------------------------------------------------------------------------
box(SPAN_C, Y_ENC, SPAN_W, H_ENC, "beh")
label(
    SPAN_C,
    Y_ENC,
    "Temporal encoder  (shared across windows)",
    r"2 $\times$ [Conv1d, kernel 3, GELU]  $\to$  tokens $\mathbf{Z}_r \in \mathbb{R}^{T \times W}$",
)
for cx in COLS:
    arrow((cx, Y_X - H_X / 2), (cx, Y_ENC + H_ENC / 2))

# --------------------------------------------------------------------------
# Context-queried cross-attention (with a sketch of the attention weights)
# --------------------------------------------------------------------------
ATT_TOP = Y_ATT + ATT_H / 2
box(SPAN_C, Y_ATT, SPAN_W, ATT_H, "fuse")
text(SPAN_C, ATT_TOP - 2.8, "Context-queried temporal cross-attention", weight="bold")
text(
    SPAN_C,
    ATT_TOP - 6.0,
    "4 heads, shared across windows;",
    size=FS_S,
    color=MUTED,
)
text(
    SPAN_C,
    ATT_TOP - 9.1,
    "the context sets the weights over time steps",
    size=FS_S,
    color=MUTED,
)

# Illustrative attention profiles: different events emphasise different steps.
profiles = (
    lambda t: 0.25 + 0.15 * math.sin(6.0 * t),
    lambda t: 0.15 + 0.85 * math.exp(-((t - 0.28) ** 2) / 0.012),
    lambda t: (
        0.12
        + 0.7 * math.exp(-((t - 0.72) ** 2) / 0.02)
        + 0.25 * math.exp(-((t - 0.2) ** 2) / 0.01)
    ),
)
for cx, prof in zip(COLS, profiles):
    cw = 1.05
    x0 = cx - N_CELLS * cw / 2
    base = Y_ATT - ATT_H / 2 + 1.8
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
    text(x0 - 1.0, base + 4.6, r"$\alpha$", size=FS + 0.5, ha="right", color="#6A52A0")
    text(
        x0 - 1.0,
        base + 0.9,
        r"$\mathbf{Z}$",
        size=FS_M,
        ha="right",
        color="#3A6EA8",
    )

for cx in COLS:
    arrow((cx, Y_ENC - H_ENC / 2), (cx, ATT_TOP))
    text(
        cx + 1.2,
        between(Y_ENC - H_ENC / 2, ATT_TOP),
        r"$\mathbf{Z}_r$",
        size=FS_M,
        ha="left",
    )

# --------------------------------------------------------------------------
# Context stream (left)
# --------------------------------------------------------------------------
box(CTX_X, Y_EXTRACT, CTX_W, H_EXTRACT, "fixed")
text(CTX_X, Y_EXTRACT + 4.6, r"Event context  $\mathbf{c}$", weight="bold")
text(CTX_X, Y_EXTRACT + 0.2, r"indicators ($d_c = 4$)  or", size=FS_S, color=MUTED)
text(CTX_X, Y_EXTRACT - 3.6, r"VLM description ($d_c = 384$)", size=FS_S, color=MUTED)

LIN_X = 24.0
box(LIN_X, Y_ENC, 22, H_ENC, "ctx")
label(
    LIN_X,
    Y_ENC,
    "Linear",
    r"$\hat{\mathbf{c}} = \mathbf{W}_c\mathbf{c} + \mathbf{b}_c$",
)
arrow((LIN_X, Y_EXTRACT - H_EXTRACT / 2), (LIN_X, Y_ENC + H_ENC / 2))

# Query = projected context + region embedding.
R_PLUS = 2.4
PLUS_Q = (LIN_X, Y_ATT)
ax.add_patch(Circle(PLUS_Q, R_PLUS, fc="white", ec=ARROW, lw=0.9, zorder=5))
text(*PLUS_Q, "+", size=FS + 2, weight="bold")
arrow((LIN_X, Y_ENC - H_ENC / 2), (LIN_X, Y_ATT + R_PLUS))
text(
    LIN_X + 1.2,
    between(Y_ENC - H_ENC / 2, Y_ATT + R_PLUS),
    r"$\hat{\mathbf{c}}$",
    size=FS + 0.5,
    ha="left",
)

EMB_X, EMB_W = 13.0, 12.5
box(EMB_X, Y_ATT, EMB_W, 9.0, "ctx")
label(EMB_X, Y_ATT, "Region", r"emb. $\mathbf{e}_r$")
arrow((EMB_X + EMB_W / 2, Y_ATT), (LIN_X - R_PLUS, Y_ATT))

arrow((LIN_X + R_PLUS, Y_ATT), (SPAN_L, Y_ATT))
text((LIN_X + R_PLUS + SPAN_L) / 2, Y_ATT + 2.4, "query", size=FS_S, color=MUTED)
text(
    (LIN_X + R_PLUS + SPAN_L) / 2,
    Y_ATT - 2.7,
    r"$\mathbf{q}_r = \hat{\mathbf{c}} + \mathbf{e}_r$",
    size=FS,
)

# Fixed lookup offset from the context.
LOOK_X, LOOK_W, LOOK_H = 27.0, 40.0, 11.0
box(LOOK_X, Y_SUM, LOOK_W, LOOK_H, "fixed")
text(LOOK_X, Y_SUM + 3.0, "Context lookup", weight="bold")
text(
    LOOK_X,
    Y_SUM - 0.5,
    r"offset $\mathbf{b}_{g(\mathbf{c})}$, not trained;",
    size=FS_S,
    color=MUTED,
)
text(LOOK_X, Y_SUM - 3.6, "fit on training folds only", size=FS_S, color=MUTED)
LOOK_LINE_X = 4.0
path_arrow(
    [
        (LOOK_LINE_X, Y_EXTRACT - H_EXTRACT / 2),
        (LOOK_LINE_X, Y_SUM),
        (LOOK_X - LOOK_W / 2, Y_SUM),
    ],
    color=STYLE["ctx"][1],
)

# --------------------------------------------------------------------------
# Inter-region Transformer, head, sum, output
# --------------------------------------------------------------------------
box(SPAN_C, Y_REG, SPAN_W, H_REG, "fuse")
label(
    SPAN_C,
    Y_REG,
    "Inter-region Transformer  (1 layer, 4 heads)",
    r"relates $[\mathbf{s}_{\mathrm{pre}}; \mathbf{s}_{\mathrm{event}}; \mathbf{s}_{\mathrm{post}}]$,"
    r" then mean over windows $\to \mathbf{h}$",
)
for cx in COLS:
    arrow((cx, Y_ATT - ATT_H / 2), (cx, Y_REG + H_REG / 2))
    text(
        cx + 1.2,
        between(Y_ATT - ATT_H / 2, Y_REG + H_REG / 2),
        r"$\mathbf{s}_r$",
        size=FS_M,
        ha="left",
    )

box(SPAN_C, Y_HEAD, 30, H_HEAD, "fuse")
label(SPAN_C, Y_HEAD, "MLP head", r"$W \to W/2 \to 2$", gap=2.9)
arrow((SPAN_C, Y_REG - H_REG / 2), (SPAN_C, Y_HEAD + H_HEAD / 2))
text(
    SPAN_C + 1.2,
    between(Y_REG - H_REG / 2, Y_HEAD + H_HEAD / 2),
    r"$\mathbf{h}$",
    size=FS_M,
    ha="left",
)

ax.add_patch(Circle((SPAN_C, Y_SUM), R_SUM, fc="white", ec=ARROW, lw=0.9, zorder=5))
text(SPAN_C, Y_SUM, "+", size=FS + 3, weight="bold")
arrow((SPAN_C, Y_HEAD - H_HEAD / 2), (SPAN_C, Y_SUM + R_SUM))
text(
    SPAN_C + 1.2,
    between(Y_HEAD - H_HEAD / 2, Y_SUM + R_SUM),
    r"$\hat{\mathbf{y}}_{\mathrm{beh}}$",
    size=FS_M,
    ha="left",
)

arrow((LOOK_X + LOOK_W / 2, Y_SUM), (SPAN_C - R_SUM, Y_SUM), color=STYLE["ctx"][1])

box(SPAN_C, Y_OUT, 54, H_OUT, "out", lw=1.1)
text(
    SPAN_C,
    Y_OUT + 2.1,
    r"$\hat{\mathbf{y}} = [\hat{y}_{\mathrm{val}},\ \hat{y}_{\mathrm{aro}}]$",
    size=FS + 1,
)
text(
    SPAN_C, Y_OUT - 2.5, "logits of the annotators' vote share", size=FS_S, color=MUTED
)
arrow((SPAN_C, Y_SUM - R_SUM), (SPAN_C, Y_OUT + H_OUT / 2))

# --------------------------------------------------------------------------
# Optional gated face stream (right)
# --------------------------------------------------------------------------
G_TOP = Y_EXTRACT + H_EXTRACT / 2 + 2.6
G_BOTTOM = Y_REG - H_REG / 2 - 2.6
group = FancyBboxPatch(
    (FACE_X - FW / 2 - 2.0, G_BOTTOM),
    FW + 4,
    G_TOP - G_BOTTOM,
    boxstyle="round,pad=0,rounding_size=2.0",
    fc="none",
    ec=STYLE["face"][1],
    lw=0.9,
    ls=(0, (3, 2)),
    zorder=1,
)
ax.add_patch(group)
text(FACE_X, G_TOP + 2.4, "Face stream", weight="bold", color=STYLE["face"][1])

box(FACE_X, Y_EXTRACT, FW, H_EXTRACT, "frozen")
text(FACE_X, Y_EXTRACT + 4.6, "EQ-Former-VIT", weight="bold")
text(FACE_X, Y_EXTRACT + 0.2, "face crop per step", size=FS_S, color=MUTED)
text(FACE_X, Y_EXTRACT - 3.6, r"quality gate $m_{r,t}$", size=FS_S, color=MUTED)

box(FACE_X, Y_ENC, FW, H_ENC, "face")
label(FACE_X, Y_ENC, "Step projection", r"LN $\to$ Linear $\to$ GELU")
arrow((FACE_X, Y_EXTRACT - H_EXTRACT / 2), (FACE_X, Y_ENC + H_ENC / 2))
text(
    FACE_X + 1.2,
    between(Y_EXTRACT - H_EXTRACT / 2, Y_ENC + H_ENC / 2),
    r"$\mathbf{x}^{f}_{r,t}$",
    size=FS_M,
    ha="left",
)

box(FACE_X, Y_ATT, FW, 9.0, "face")
label(FACE_X, Y_ATT, "Gated mean", r"per window $\to \bar{\mathbf{u}}_r$")
arrow((FACE_X, Y_ENC - H_ENC / 2), (FACE_X, Y_ATT + 4.5))
text(
    FACE_X + 1.2,
    between(Y_ENC - H_ENC / 2, Y_ATT + 4.5),
    r"$\mathbf{u}_{r,t} \in \mathbb{R}^{32}$",
    size=FS_M,
    ha="left",
)

box(FACE_X, Y_REG, FW, H_REG, "face")
label(FACE_X, Y_REG, r"Face head $\psi$", "zero-initialised")
arrow((FACE_X, Y_ATT - 4.5), (FACE_X, Y_REG + H_REG / 2))

path_arrow(
    [
        (FACE_X, Y_REG - H_REG / 2),
        (FACE_X, Y_SUM),
        (SPAN_C + R_SUM, Y_SUM),
    ],
    color=STYLE["face"][1],
)
text(
    FACE_X - 2.0,
    Y_SUM + 2.3,
    r"$\mathbf{\Delta}_{\mathrm{face}}$ (zero if no valid face)",
    size=FS_S,
    ha="right",
    color=STYLE["face"][1],
)

# --------------------------------------------------------------------------
# Legend
# --------------------------------------------------------------------------
LX, LY = 124.0, 14.6
items = (
    ("frozen", "frozen, pretrained"),
    (("beh", "ctx", "fuse", "face"), "trainable (colour = stream)"),
    ("fixed", "data or fixed, not trained"),
)
for i, (kind, name) in enumerate(items):
    y = LY - i * 4.0
    if isinstance(kind, tuple):
        # One swatch per stream, filling the same width as a single swatch.
        sw = 5.5 / len(kind)
        for k, stream in enumerate(kind):
            box(LX - 2.75 + (k + 0.5) * sw, y, sw - 0.35, 3.2, stream, lw=0.8, r=0.4)
    else:
        box(LX, y, 5.5, 3.2, kind, lw=0.8, r=0.7)
    text(LX + 4.2, y, name, size=FS_S, ha="left", color=MUTED)
y = LY - 3 * 4.0
ax.plot([LX - 2.75, LX + 2.75], [y, y], color=STYLE["face"][1], lw=0.9, ls=(0, (3, 2)))
text(LX + 4.2, y, "optional", size=FS_S, ha="left", color=MUTED)

for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
    fig.savefig(OUT.with_suffix(f".{ext}"), **kw)
print("wrote", ", ".join(OUT.with_suffix(f".{e}").name for e in ("pdf", "svg", "png")))
