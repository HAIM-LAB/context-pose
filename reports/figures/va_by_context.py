"""Valence-arousal of the ContextPose clips by event and by behaviour.

Clips: the 153 scored by the full model (pose + annotation + VLM + face) in
regression form (checkpoints/streams_reg/T_PAF_full). Human: the annotators'
consensus on [-1, 1]. Model: each clip's out-of-fold prediction, averaged over
the 5 seeds (not clipped to [-1, 1]). A clip with several event indicators
appears under each of them.

Writes event_va, behaviour_va (human row above the model row) and
event_behaviour (clip counts and mean V/A per behaviour x event cell).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "checkpoints/streams_reg/T_PAF_full"
HERE = Path(__file__).parent
EVENTS = (
    "Social interaction",
    "Auditory event",
    "Activity / object change",
    "Visual / media event",
    "No event",
)
BEHAVIOURS = {
    "armflapping": "Arm flapping",
    "spinning": "Spinning",
    "headbanging": "Head banging",
    "handaction": "Hand action",
}
SOURCES = {"human": ("Human consensus", "#3A6EA8"), "model": ("ContextPose", "#D9662B")}
INK, MUTED = "#222222", "#8A8A8A"
LIMIT = 1.5
MIN_CELL = 3  # cells with fewer clips are shown without a mean


def load() -> dict:
    """Per clip: human and model [valence, arousal], 5 event flags, behaviour."""
    seeds = [
        {p["path"]: p for p in json.load(open(path))["predictions"]}
        for path in sorted(RUN.glob("seed*/cv_report.json"))
    ]
    paths = sorted(seeds[0])
    assert all(sorted(s) == paths for s in seeds), "seeds scored different clips"
    events = np.array([np.load(p)["context"] for p in paths]).astype(bool)
    return {
        "human": np.array([seeds[0][p]["target"] for p in paths]),
        "model": np.mean([[s[p]["pred"] for p in paths] for s in seeds], 0),
        "events": np.c_[events, ~events.any(1)],  # no event: all four off
        "behaviour": np.array([seeds[0][p]["category"] for p in paths]),
    }


def circle(ax, va: np.ndarray, title: str, colour: str) -> None:
    ax.add_patch(Circle((0, 0), 1, fill=False, ls=":", lw=0.8, color=MUTED))
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    for x, y, name in [
        (-0.75, 1.3, "distressed"),
        (0.75, 1.3, "excited"),
        (-0.75, -1.3, "sad / bored"),
        (0.75, -1.3, "calm / content"),
    ]:
        ax.text(x, y, name, ha="center", va="center", fontsize=7, color=MUTED, style="italic")  # fmt: skip
    ax.scatter(va[:, 0], va[:, 1], s=14, color=colour, alpha=0.45, lw=0)
    mean = va.mean(0)
    ax.scatter(*mean, s=90, color=colour, edgecolor=INK, lw=1.2, zorder=3)
    ax.annotate(
        f"mean ({mean[0]:+.2f}, {mean[1]:+.2f})",
        mean,
        xytext=(0, -10),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=7.5,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
    )
    pleasant, aroused = va[:, 0] > 0, va[:, 1] > 0
    counts = {
        "V>0, A>0": pleasant & aroused,
        "V≤0, A>0": ~pleasant & aroused,
        "V>0, A≤0": pleasant & ~aroused,
        "V≤0, A≤0": ~pleasant & ~aroused,
    }
    cells = [f"{k}: {m.sum()} ({100 * m.mean():.0f}%)" for k, m in counts.items()]
    ax.set_title(f"{title}  (n = {len(va)})", fontsize=9.5, color=INK)
    ax.text(
        0.5,
        -0.2,
        "   ".join(cells[:2]) + "\n" + "   ".join(cells[2:]),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color=INK,
    )
    ax.set(xlim=(-LIMIT, LIMIT), ylim=(-LIMIT, LIMIT), aspect="equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(labelsize=7, colors=INK)
    ax.set_xlabel("Valence (unpleasant → pleasant)", fontsize=7.5)
    for side in ax.spines.values():
        side.set_color(MUTED)


def grid(data: dict, groups: list[tuple[str, np.ndarray]], subject: str, out: str) -> None:  # fmt: skip
    """Human row above model row; one column per group (first is all clips)."""
    fig, axes = plt.subplots(2, len(groups), figsize=(3.1 * len(groups), 7.6))
    for row, (source, (name, colour)) in zip(axes, SOURCES.items()):
        for ax, (title, mask) in zip(row, groups):
            circle(ax, data[source][mask], title, colour)
        row[0].set_ylabel(f"{name}\nArousal (calm → aroused)", fontsize=8.5)
    fig.suptitle(
        f"Valence–arousal by {subject}: human consensus (top) vs ContextPose "
        "out-of-fold predictions (bottom)",
        fontsize=11,
    )
    fig.subplots_adjust(hspace=0.5, wspace=0.28, top=0.9, bottom=0.1)
    save(fig, out)


def crosstab(data: dict) -> None:
    """Behaviour x event: clip counts, then mean valence and arousal per cell."""
    rows = list(BEHAVIOURS)
    masks = np.array([[(data["behaviour"] == b) & e for e in data["events"].T] for b in rows])  # fmt: skip
    counts = masks.sum(-1)
    panels = [("Clips", None, None)] + [
        (f"{SOURCES[s][0]}: mean {t}", s, i)
        for s in SOURCES
        for i, t in enumerate(("valence", "arousal"))
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.4), sharey=True)  # fmt: skip
    for ax, (title, source, index) in zip(axes, panels):
        if source is None:
            values = counts.astype(float)
            counts_image = ax.imshow(values, cmap="Blues", vmin=0, vmax=counts.max())
            labels = [[str(n) for n in row] for row in counts]
        else:
            values = np.full(counts.shape, np.nan)
            for r in range(len(rows)):
                for c in range(len(EVENTS)):
                    if counts[r, c] >= MIN_CELL:
                        values[r, c] = data[source][masks[r, c], index].mean()
            # Blue = pleasant / aroused, red = unpleasant / calm, white = 0.
            mean_image = ax.imshow(values, cmap="RdBu", vmin=-1, vmax=1)
            labels = [
                [f"{v:+.2f}" if np.isfinite(v) else f"n={n}" for v, n in zip(vr, nr)]
                for vr, nr in zip(values, counts)
            ]
            ax.set_facecolor("#EEEEEE")
        for (r, c), text in np.ndenumerate(np.array(labels, dtype=object)):
            v = values[r, c]
            dark = np.isfinite(v) and (
                v > 0.6 * counts.max() if source is None else abs(v) > 0.6
            )
            ax.text(c, r, text, ha="center", va="center", fontsize=8,
                    color="white" if dark else (INK if np.isfinite(v) else MUTED))  # fmt: skip
        ax.set_xticks(range(len(EVENTS)), ["Social", "Auditory", "Activity/\nobject", "Visual/\nmedia", "None"], fontsize=7.5)  # fmt: skip
        ax.set_yticks(range(len(rows)), [BEHAVIOURS[b] for b in rows], fontsize=8)
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    bar = fig.colorbar(counts_image, ax=axes[0], fraction=0.046, pad=0.04)
    bar.ax.tick_params(labelsize=7)
    bar = fig.colorbar(mean_image, ax=axes[1:].tolist(), fraction=0.012, pad=0.015)
    bar.ax.tick_params(labelsize=7)
    bar.set_label("mean (V: unpleasant → pleasant, A: calm → aroused)", fontsize=7.5)
    fig.suptitle(
        "Events within each behaviour: clip counts and mean valence / arousal "
        f"(cells under {MIN_CELL} clips show only n)",
        fontsize=11,
    )
    save(fig, "event_behaviour")


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = load()
    every = np.ones(len(data["human"]), bool)
    events = [("All clips", every)] + list(zip(EVENTS, data["events"].T))
    behaviours = [("All clips", every)] + [
        (name, data["behaviour"] == key) for key, name in BEHAVIOURS.items()
    ]
    grid(data, events, "event", "event_va")
    grid(data, behaviours, "behaviour", "behaviour_va")
    crosstab(data)


if __name__ == "__main__":
    main()
