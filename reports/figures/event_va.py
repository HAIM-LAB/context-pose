"""Consensus valence-arousal of the context clips, one panel per event.

Reads the three-annotator consensus labels and event indicators from the
`Final_detailed` sheet of Final_Annotation_v2.xlsx (the sheet `prepare.py`
reads). Each panel shows the clips whose annotators marked that event, over all
clips in grey, with the event's mean and quadrant counts. A clip with several
events appears in each of their panels.

    uv run python reports/figures/event_va.py PATH/TO/Final_Annotation_v2.xlsx
    uv run python reports/figures/event_va.py PATH/TO/Final_Annotation_v2.xlsx --by combination

`--by combination` draws one panel per indicator combination with at least
`--min-clips` clips (the groups of the event-annotation lookup) instead.
Writes event_va[_combination].{pdf,png} next to this script unless --output.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl

logging.getLogger("fontTools").setLevel(logging.ERROR)  # quiet PDF font subsetting

EVENTS = {
    "social_interaction": "Social interaction",
    "auditory_event": "Auditory event",
    "activity_object_change": "Activity / object change",
    "visual_media_event": "Visual / media event",
}
SHORT = {
    "social_interaction": "social",
    "auditory_event": "auditory",
    "activity_object_change": "object change",
    "visual_media_event": "visual/media",
}
COLOR = "#2f5d9e"
BACKGROUND = "#c9c9c9"
TEXT = "#333333"
JITTER = 0.025  # labels sit on a few values; spread coincident points slightly


def load(path: Path) -> list[dict]:
    """Clips with numeric consensus valence and arousal and 0/1 indicators."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = workbook["Final_detailed"].iter_rows(values_only=True)
    header = next(rows)
    clips = []
    for row in rows:
        record = dict(zip(header, row))
        if not record.get("clip_name"):
            continue
        try:
            clip = {t: float(record[t]) for t in ("valence", "arousal")}
            clip.update({k: int(record[k] or 0) for k in EVENTS})
        except (TypeError, ValueError):
            continue  # unknown label
        clip["name"] = record["clip_name"]
        clips.append(clip)
    return clips


def quadrant_counts(v: np.ndarray, a: np.ndarray) -> str:
    n = len(v)
    on_axis = (v == 0) | (a == 0)
    cells = [
        ("pleasant · aroused", (v > 0) & (a > 0)),
        ("unpleasant · aroused", (v < 0) & (a > 0)),
        ("unpleasant · calm", (v < 0) & (a < 0)),
        ("pleasant · calm", (v > 0) & (a < 0)),
    ]
    parts = [f"{name}: {mask.sum()} ({100 * mask.sum() / n:.0f}%)" for name, mask in cells]
    return (
        f"{parts[0]}   {parts[1]}\n{parts[2]}   {parts[3]}\n"
        f"on an axis (valence or arousal exactly 0): {on_axis.sum()}"
    )


def circumplex(ax) -> None:
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls="--", lw=0.8, color="#aaaaaa"))
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
    ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=0)
    for x, y, name in [(-0.72, 0.72, "distressed"), (0.72, 0.72, "excited"),
                       (-0.72, -0.72, "sad / bored"), (0.72, -0.72, "calm / content")]:  # fmt: skip
        ax.text(x, y, name, ha="center", va="center", fontsize=8.5,
                style="italic", color="#999999")  # fmt: skip
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(labelsize=8.5)


def panel(ax, title, v, a, all_v, all_a, overall, rng, background=True) -> None:
    circumplex(ax)
    jitter = lambda x: x + rng.uniform(-JITTER, JITTER, len(x))  # noqa: E731
    if background:
        ax.scatter(jitter(all_v), jitter(all_a), s=10, color=BACKGROUND, lw=0, zorder=1)
        ax.scatter(*overall, s=90, facecolor="white", edgecolor="#777777", lw=1.4, zorder=3)
    ax.scatter(jitter(v), jitter(a), s=18, color=COLOR, alpha=0.6, lw=0, zorder=2)
    mean = (v.mean(), a.mean())
    ax.scatter(*mean, s=150, color=COLOR, edgecolor="black", lw=1.4, zorder=4)
    ax.annotate(f"mean ({mean[0]:+.2f}, {mean[1]:+.2f})", mean, xytext=(8, 8),
                textcoords="offset points", fontsize=8.5, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))  # fmt: skip
    ax.set_title(f"{title}  (n = {len(v)})", fontsize=10.5, color=TEXT)
    ax.text(0.5, -0.2, quadrant_counts(v, a), transform=ax.transAxes, ha="center",
            va="top", fontsize=7.5, color=TEXT)  # fmt: skip


def groups(clips: list[dict], by: str, min_clips: int) -> list[tuple[str, list[dict]]]:
    if by == "event":
        found = [(name, [c for c in clips if c[k]]) for k, name in EVENTS.items()]
        found.append(("No event", [c for c in clips if not any(c[k] for k in EVENTS)]))
        return [(name, members) for name, members in found if members]
    combos = {}
    for clip in clips:
        key = tuple(k for k in EVENTS if clip[k])
        combos.setdefault(key, []).append(clip)
    ordered = sorted(combos.items(), key=lambda item: -len(item[1]))
    return [
        (" + ".join(SHORT[k] for k in key) or "No event", members)
        for key, members in ordered
        if len(members) >= min_clips
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("workbook", type=Path, help="Final_Annotation_v2.xlsx")
    parser.add_argument("--by", choices=["event", "combination"], default="event")
    parser.add_argument("--min-clips", type=int, default=3,
                        help="smallest combination to draw (--by combination)")  # fmt: skip
    parser.add_argument("--output", type=Path, help="output path without extension")
    args = parser.parse_args(argv)

    clips = load(args.workbook)
    if not clips:
        parser.error(f"no labelled clips in {args.workbook}")
    all_v = np.array([c["valence"] for c in clips])
    all_a = np.array([c["arousal"] for c in clips])
    overall = (all_v.mean(), all_a.mean())
    found = [("All clips", clips)] + groups(clips, args.by, args.min_clips)

    columns = 3
    rows = -(-len(found) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 5.5 * rows),
                             squeeze=False)  # fmt: skip
    rng = np.random.default_rng(0)
    for index, (ax, (name, members)) in enumerate(zip(axes.flat, found)):
        v = np.array([c["valence"] for c in members])
        a = np.array([c["arousal"] for c in members])
        panel(ax, name, v, a, all_v, all_a, overall, rng, background=index > 0)
    for ax in axes.flat[len(found):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Arousal (calm → aroused)", fontsize=9.5)
    for ax in axes.flat[: len(found)]:
        ax.set_xlabel("Valence (unpleasant → pleasant)", fontsize=9.5)

    kind = "event" if args.by == "event" else "event combination"
    fig.suptitle(
        f"Consensus valence–arousal of the context clips by {kind}  (all clips: n = {len(clips)})",
        fontsize=13, color=TEXT,
    )  # fmt: skip
    fig.text(
        0.5, 0.005,
        f"Blue: clips in the panel (big dot = their mean).  Grey: all clips "
        f"(hollow dot = overall mean {overall[0]:+.2f}, {overall[1]:+.2f}).  "
        "Points are jittered slightly.",
        ha="center", va="bottom", fontsize=9, color="#666666",
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.035, 1, 0.96), h_pad=4)

    suffix = "" if args.by == "event" else "_combination"
    out = args.output or Path(__file__).with_name(f"event_va{suffix}")
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    print(f"wrote {out}.pdf and .png ({len(clips)} clips, {len(found)} panels)")
    for name, members in found:
        v = np.mean([c["valence"] for c in members])
        a = np.mean([c["arousal"] for c in members])
        print(f"  {name:<40} n={len(members):<4} mean valence {v:+.2f}  arousal {a:+.2f}")


if __name__ == "__main__":
    main()
