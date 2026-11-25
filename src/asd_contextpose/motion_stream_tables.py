"""Motion streams: a motion descriptor and motion-guided VST as model inputs.

Writes two tables in the layout of `video_stream_tables`, from the 5-seed
cross-validation runs already on disk: binary calls against the final
annotation (`--objective final` runs) and regression of the final annotation
(regression-objective runs). The new rows are the `checkpoints/motion/fin`
and `checkpoints/motion/reg` runs (`--stream motion:53`,
`--stream vst_motion:1024`, features from `motion_features`); the reference
rows are runs of tab:eavf and tab:video. Metrics, ensembling and references
are those of `eclip_stream_tables`.

    uv run python -m asd_contextpose.motion_stream_tables \
        --output reports/motion_streams.tex
"""

import argparse
import json
from pathlib import Path

from asd_contextpose.eclip_stream_tables import (
    ROOT,
    RUNS,
    TARGETS,
    annotators,
    binary,
    document,
    load,
    number,
    percent,
    regression,
)
from asd_contextpose.motion_features import MOTION_DIM
from asd_contextpose.video_stream_tables import label, rows_with_groups, video

# S = Video Swin (VST), H = HOG-HOF, M = motion descriptor, G = motion-guided VST
COLUMNS = "EAVFSHMG"
HEADERS = ("E", "A", "V", "F", "VST", "H", "M", "VSTm")
NAMES = {
    "E": "EmotionCLIP",
    "A": "annotation",
    "V": "VLM",
    "F": "face",
    "S": "VST",
    "H": "HOG-HOF",
    "M": "motion",
    "G": "motion-guided VST",
}


def motion(name: str) -> tuple[str, str]:
    return f"motion/fin/{name}", f"motion/reg/{name}"


REFERENCE = "Reference rows without motion streams"
ALONE = "Motion streams alone and with other video streams"
BEST = "Added to the best rows"
# (group, streams, label suffix, (final run, regression run))
ROWS = [
    (REFERENCE, "AF", "", RUNS["AF"]),
    (REFERENCE, "EAF", "", RUNS["EAF"]),
    (REFERENCE, "AVF", "", RUNS["AVF"]),
    (REFERENCE, "S", "", video("VST")),
    (REFERENCE, "H", "", video("H")),
    (REFERENCE, "SH", "", video("VST_H")),
    (REFERENCE, "AFS", "", video("VST_AF")),
    (REFERENCE, "EAVFSH", ", stream dropout 0.3", video("ALL_sdrop")),
    (ALONE, "M", "", motion("M")),
    (ALONE, "G", "", motion("SM")),
    (ALONE, "MG", "", motion("SM_M")),
    (ALONE, "SHM", "", motion("VST_H_M")),
    (ALONE, "EM", "", motion("E_M")),
    (BEST, "AFM", "", motion("M_AF")),
    (BEST, "AFG", "", motion("SM_AF")),
    (BEST, "EAFM", "", motion("E_M_AF")),
    (BEST, "AVFM", "", motion("M_AVF")),
    ("All eight streams", COLUMNS, ", stream dropout 0.3", motion("ALL_sdrop")),
]
STREAMS = (
    r"\emph{Motion streams} (\code{asd\_contextpose.motion\_features}): at each of the 32 steps of the "
    r"pre-event, event and post-event windows, a 32-frame window at 15 fps (about 2 s, twice the VST "
    r"window) centred on the step is cropped to the child as for the video streams.",
    r"\begin{itemize}",
    rf"  \item M: a hand-designed motion descriptor ({MOTION_DIM}-d). Farneback optical flow on a 96-pixel grey crop, with each frame pair's median flow (camera motion and whole-body drift) removed, gives the log motion energy per cell (4$\times$4) and a time-exposure image (per-pixel temporal standard deviation, per cell); statistics of the motion-energy time course (log mean, spread and peak, and the share of local peaks); its periodicity (the energy spectrum's share in the bands 0.5--1, 1--2, 2--3, 3--4, 4--5.5 and 5.5--7.5\,Hz, and the dominant frequency of the horizontal and vertical flow with its share of the power); the flow's direction (log horizontal and vertical energy and the vertical share); its rotation and expansion (mean curl, mean $|$curl$|$ and mean divergence); and the log removed camera motion.",
    r"  \item VSTm: the frozen Video Swin Transformer of the VST stream (1024-d) on 16 of the 32 frames, picked by motion after MGSampler (Zhi et al., 2021): at equal steps of the cumulative motion energy, so fast movement gets more frames and still stretches fewer, in temporal order.",
    r"\end{itemize}",
)  # fmt: skip


def binary_table(calls: list[dict], people: dict) -> str:
    keys = ("accuracy", "balanced_accuracy", "f1", "mcc", "auc")
    models = [r for _, r in calls]
    best = {
        (t, k): max(r[t]["model"][k] for r in models if r[t]["model"][k] is not None)
        for t in TARGETS
        for k in keys
    }
    first = models[0]

    def cells(m: dict, target: str, is_model: bool) -> str:
        b = (lambda k: best[(target, k)]) if is_model else (lambda k: None)
        ci = m.get("balanced_accuracy_95ci")
        interval = rf"\ci{{{round(100 * ci[0])}}}{{{round(100 * ci[1])}}}" if ci else ""
        return " & ".join([
            percent(m["accuracy"], b("accuracy")),
            percent(m["balanced_accuracy"], b("balanced_accuracy")) + interval,
            number(m["f1"], 2, b("f1")),
            number(m["mcc"], 2, b("mcc")),
            number(m["auc"], 2, b("auc")),
        ])  # fmt: skip

    def person(target: str) -> str:
        p = people[target]
        return f"{percent(p['accuracy'])} & {percent(p['balanced_accuracy'])} & -- & -- & --"

    n = {t: first[t]["n"] for t in TARGETS}
    pos = {t: first[t]["positives"] for t in TARGETS}
    span = len(COLUMNS) + 1
    lookup = r" & $\checkmark$" + " &" * (len(COLUMNS) - 1)
    rows = [
        (row, f"{cells(r['valence']['model'], 'valence', True)} & {cells(r['arousal']['model'], 'arousal', True)}")  # fmt: skip
        for row, r in calls
    ]
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{\textbf{Motion streams on the binary valence and arousal calls} (cross-validated, "
        r"5-seed ensembles). E = EmotionCLIP, A = event annotation, V = VLM scene description, F = face, "
        r"VST = Video Swin Transformer features, H = HOG-HOF features, M = motion descriptor, VSTm = "
        r"motion-guided VST; ticks mark each row's inputs and pose is never an input. \emph{Reference "
        r"rows}: earlier runs without motion streams (the EmotionCLIP and video stream tables), for "
        r"comparison. \emph{Alone and with other video streams}: motion features with no context stream, "
        r"together, next to VST and HOG-HOF, and next to EmotionCLIP. \emph{Added to the best rows}: M or "
        r"VSTm added to A+F, E+A+F and A+V+F. \emph{All eight streams}: every stream at once, with stream "
        r"dropout 0.3 (whole streams hidden at random in training). Rows with A use the event-text "
        r"offset; rows with V use the VLM description as the context query.}",
        r"\label{tab:motion}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{'c' * len(COLUMNS)}l{'c' * 10}}}",
        r"\toprule",
        rf"\multicolumn{{{span}}}{{l}}{{}} & \multicolumn{{5}}{{c}}{{Valence: positive vs not ($n{{=}}{n['valence']}$)}} "
        rf"& \multicolumn{{5}}{{c}}{{Arousal: high vs low ($n{{=}}{n['arousal']}$)}} \\",
        rf"\cmidrule(lr){{{span + 1}-{span + 5}}}\cmidrule(lr){{{span + 6}-{span + 10}}}",
        " & ".join(HEADERS) + r" & Model & Acc. & Bal. acc. & F1 & MCC & AUC & Acc. & Bal. acc. & F1 & MCC & AUC \\",
        r"\midrule",
        rf"\multicolumn{{{span}}}{{l}}{{Annotators (each vs the other two)}} & {person('valence')} & {person('arousal')} \\",
        rf"\multicolumn{{{span}}}{{l}}{{Always majority}} & {cells(first['valence']['majority'], 'valence', False)} "
        rf"& {cells(first['arousal']['majority'], 'arousal', False)} \\",
        rf"{lookup} Annotation lookup (no network) & {cells(first['valence']['lookup'], 'valence', False)} "
        rf"& {cells(first['arousal']['lookup'], 'arousal', False)} \\",
        *rows_with_groups(rows, 10, COLUMNS, NAMES),
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\tabledesc{Each row is a ContextPose model given the ticked input streams. The reference rows are "
        r"runs of the pose-free stream ablation (\code{tab:eavf} in eclip\_streams.tex) and of the video "
        r"streams (\code{tab:video} in video\_streams.tex); the other rows add one or both motion streams.",
        *STREAMS,
        r"Each motion stream enters with a child-tracked flag, gets its own temporal encoder and modality embedding, and its tokens join those the context query attends over (\code{--stream}). Stream dropout hides each stream of a training sample with probability 0.3.",
        r"\emph{Offsets and protocol:} as in \code{tab:eavf}: rows with A start from the event-annotation lookup and learn a correction, rows without A from a constant fitted on the training folds; 153 development clips, video-grouped 5-fold cross-validation with duplicate videos merged, cross-entropy against the final annotation's binary classes, identical training settings for every row.",
        rf"\emph{{Targets and metrics:}} as in \code{{tab:eavf}}: classes of the final annotation, positive valence / high arousal when its score is $> 0$ (valence: {n['valence']} clips, {pos['valence']} positive; arousal: {n['arousal']} clips, {pos['arousal']} high); the ensemble's call is the majority of the 5 seeds' calls and AUC uses the mean seed score; balanced accuracy with 95\% participant-bootstrap intervals. Bold marks the best model in each column.}}",
        r"\end{table}",
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def regression_table(reg: list[dict]) -> str:
    keys = (("spearman", max), ("ccc", max), ("mae", min), ("ap", max), ("mcc", max))
    models = [r["model"] for _, r in reg]
    best = {
        (t, k): pick(m[t][k] for m in models if m[t][k] is not None)
        for t in TARGETS
        for k, pick in keys
    }
    best["map"] = max(m["map"] for m in models)
    first = reg[0][1]

    def cells(m: dict, is_model: bool) -> str:
        parts = []
        for t in TARGETS:
            for k, _ in keys:
                digits = 3 if k == "mae" else 2
                parts.append(number(m[t][k], digits, best[(t, k)] if is_model else None))
        parts.append(number(m["map"], 3, best["map"] if is_model else None))
        return " & ".join(parts)

    span = len(COLUMNS) + 1
    lookup = r" & $\checkmark$" + " &" * (len(COLUMNS) - 1)
    rows = [(row, cells(r["model"], True)) for row, r in reg]
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{\textbf{Motion streams as regression of the final valence and arousal annotation} "
        r"(cross-validated, 5 seeds). Streams and row groups as in Table~\ref{tab:motion}. Higher Spearman, "
        r"CCC, AP, MCC and mAP and lower MAE are better.}",
        r"\label{tab:motion_reg}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{'c' * len(COLUMNS)}l{'c' * 11}}}",
        r"\toprule",
        rf"\multicolumn{{{span}}}{{l}}{{}} & \multicolumn{{5}}{{c}}{{Valence}} & \multicolumn{{5}}{{c}}{{Arousal}} & \\",
        rf"\cmidrule(lr){{{span + 1}-{span + 5}}}\cmidrule(lr){{{span + 6}-{span + 10}}}",
        " & ".join(HEADERS) + r" & Model & Spearman & CCC & MAE & AP & MCC & Spearman & CCC & MAE & AP & MCC & mAP \\",
        r"\midrule",
        rf"\multicolumn{{{span}}}{{l}}{{Overall median}} & {cells(first['median'], False)} \\",
        rf"{lookup} Annotation lookup (no network) & {cells(first['lookup'], False)} \\",
        *rows_with_groups(rows, 11, COLUMNS, NAMES),
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\tabledesc{Each row is the model of the same row in Table~\ref{tab:motion}, trained to regress the "
        r"final annotation's score ($[-1, 1]$ scale) instead of classifying its sign; "
        r"predictions are averaged over 5 seeds. Reference rows are runs of \code{tab:eavf\_reg} in "
        r"eclip\_streams.tex and \code{tab:video\_reg} in video\_streams.tex. "
        rf"{first['n']} development clips; protocol, offsets and metrics as in "
        r"\code{tab:eavf\_reg} (AP ranks the clips with a positive final annotation above the rest; MCC "
        r"compares the signs of prediction and label). Bold marks the best model in each column.}",
        r"\end{table}",
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--output", type=Path, default=ROOT / "reports/motion_streams.tex")
    parser.add_argument("--json", type=Path, help="also write the metrics as JSON")
    args = parser.parse_args(argv)

    calls, reg = [], []
    for row in ROWS:
        final_run, reg_run = row[3]
        calls.append((row, binary(load(final_run))))
        reg.append((row, regression(load(reg_run))))
        v, a = calls[-1][1]["valence"]["model"], calls[-1][1]["arousal"]["model"]
        r = reg[-1][1]["model"]
        name = label(row[1], row[2], COLUMNS, NAMES)
        print(
            f"{name:55s} bal V {v['balanced_accuracy']:.3f} A {a['balanced_accuracy']:.3f}"
            f" AUC V {v['auc']:.2f} A {a['auc']:.2f} | rho V {r['valence']['spearman']:.2f}"
            f" A {r['arousal']['spearman']:.2f} mAP {r['map']:.3f}"
        )  # fmt: skip
    people = annotators(RUNS["A"][0])
    body = binary_table(calls, people) + "\n" + regression_table(reg)
    text = document(body)
    text = text.replace("Stream Ablation without Pose", "Motion Streams")
    text = text.replace("eclip_stream_tables`", "motion_stream_tables`")
    args.output.write_text(text)
    print(f"wrote {args.output}")
    if args.json:
        payload = {
            "calls": {label(r[1], r[2], COLUMNS, NAMES): c for r, c in calls},
            "regression": {label(r[1], r[2], COLUMNS, NAMES): g for r, g in reg},
            "annotators": people,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
