"""Video streams: Video Swin Transformer (VST) and HOG-HOF features as model inputs.

Writes two tables in the layout of `eclip_stream_tables`, from the 5-seed
cross-validation runs already on disk: binary calls against the final
annotation (`--objective final` runs) and regression of the final annotation
(regression-objective runs). The new rows are the `checkpoints/final/video`
and `checkpoints/video/reg` runs (`--stream vst:1024`, `--stream hoghof:274`,
features from `video_features`); the reference rows without video streams are
the runs of tab:eavf and tab:eavf_reg. Metrics, ensembling and references are
those of `eclip_stream_tables`.

    uv run python -m asd_contextpose.video_stream_tables \
        --output reports/video_streams.tex
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

COLUMNS = ("E", "A", "V", "F", "S", "H")  # S = Video Swin (VST), H = HOG-HOF
HEADERS = ("E", "A", "V", "F", "VST", "H")
NAMES = {
    "E": "EmotionCLIP",
    "A": "annotation",
    "V": "VLM",
    "F": "face",
    "S": "VST",
    "H": "HOG-HOF",
}


def video(name: str) -> tuple[str, str]:
    return f"final/video/{name}", f"video/reg/{name}"


# (group, streams, label suffix, (votes run, regression run))
ROWS = [
    ("Reference rows without video streams", "E", "", RUNS["E"]),
    ("Reference rows without video streams", "A", "", RUNS["A"]),
    ("Reference rows without video streams", "V", "", RUNS["V"]),
    ("Reference rows without video streams", "AF", "", RUNS["AF"]),
    ("Reference rows without video streams", "EAF", "", RUNS["EAF"]),
    ("Reference rows without video streams", "AVF", "", RUNS["AVF"]),
    ("Reference rows without video streams", "EAVF", "", RUNS["EAVF"]),
    ("Video streams alone and with EmotionCLIP", "S", "", video("VST")),
    ("Video streams alone and with EmotionCLIP", "H", "", video("H")),
    ("Video streams alone and with EmotionCLIP", "SH", "", video("VST_H")),
    ("Video streams alone and with EmotionCLIP", "ES", "", video("E_VST")),
    ("Added to one stream", "AS", "", video("VST_A")),
    ("Added to one stream", "FS", "", video("VST_F")),
    ("Added to one stream", "VS", "", video("VST_V")),
    ("Added to one stream", "VH", "", video("H_V")),
    ("Added to the best rows", "AFS", "", video("VST_AF")),
    ("Added to the best rows", "AFH", "", video("H_AF")),
    ("Added to the best rows", "AFSH", "", video("VST_H_AF")),
    ("Added to the best rows", "EAFS", "", video("E_VST_AF")),
    ("Added to the best rows", "EAFH", "", video("E_H_AF")),
    ("Added to the best rows", "AVFS", "", video("VST_AVF")),
    ("All six streams", "EAVFSH", "", video("ALL")),
    ("All six streams", "EAVFSH", ", stream dropout 0.3", video("ALL_sdrop")),
]


def ticks(streams: str) -> str:
    return " & ".join(r"$\checkmark$" if s in streams else "" for s in COLUMNS)


def label(streams: str, suffix: str) -> str:
    if len(streams) == len(COLUMNS):
        return "All six" + suffix
    text = " + ".join(NAMES[s] for s in COLUMNS if s in streams)
    return text[0].upper() + text[1:] + (" only" if len(streams) == 1 else "") + suffix


def rows_with_groups(results: list[dict], metrics: int) -> list[str]:
    """Table rows, with a rule and a group heading wherever the group changes."""
    width = len(COLUMNS) + 1 + metrics
    lines, previous = [], None
    for (group, streams, suffix, _), cells in results:
        if group != previous:
            lines += [r"\midrule", rf"\multicolumn{{{width}}}{{l}}{{\emph{{{group}}}}} \\"]  # fmt: skip
            previous = group
        lines.append(f"{ticks(streams)} & {label(streams, suffix)} & {cells} \\\\")
    return lines


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
    rows = [
        (row, f"{cells(r['valence']['model'], 'valence', True)} & {cells(r['arousal']['model'], 'arousal', True)}")  # fmt: skip
        for row, r in calls
    ]
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{\textbf{Video streams on the binary valence and arousal calls} (cross-validated, "
        r"5-seed ensembles). E = EmotionCLIP, A = event annotation, V = VLM scene description, F = face, "
        r"VST = Video Swin Transformer features, H = HOG-HOF features; ticks mark each row's inputs and "
        r"pose is never an input. \emph{Reference rows}: earlier runs without video streams "
        r"(the EmotionCLIP stream tables), for comparison. \emph{Alone and with EmotionCLIP}: video features with "
        r"no other stream, and the VST next to EmotionCLIP. \emph{Added to one stream}: VST or HOG-HOF "
        r"added to a single existing stream. \emph{Added to the best rows}: VST and/or HOG-HOF added to "
        r"A+F (best valence), E+A+F (best AUC) and A+V+F. \emph{All six streams}: every stream at once, "
        r"without and with stream dropout 0.3 (whole streams hidden at random in training). Rows with A "
        r"use the event-text offset; rows with V use the VLM description as the context query.}",
        r"\label{tab:video}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{cccccclcccccccccc}",
        r"\toprule",
        rf"\multicolumn{{{span}}}{{l}}{{}} & \multicolumn{{5}}{{c}}{{Valence: positive vs not ($n{{=}}{n['valence']}$)}} "
        rf"& \multicolumn{{5}}{{c}}{{Arousal: high vs low ($n{{=}}{n['arousal']}$)}} \\",
        rf"\cmidrule(lr){{{span + 1}-{span + 5}}}\cmidrule(lr){{{span + 6}-{span + 10}}}",
        " & ".join(HEADERS) + r" & Model & Acc. & Bal. acc. & F1 & MCC & AUC & Acc. & Bal. acc. & F1 & MCC & AUC \\",
        r"\midrule",
        rf"\multicolumn{{{span}}}{{l}}{{Annotators (each vs the other two)}} & {person('valence')} & {person('arousal')} \\",
        rf"\multicolumn{{{span}}}{{l}}{{Always majority}} & {cells(first['valence']['majority'], 'valence', False)} "
        rf"& {cells(first['arousal']['majority'], 'arousal', False)} \\",
        rf" & $\checkmark$ & & & & & Annotation lookup (no network) & {cells(first['valence']['lookup'], 'valence', False)} "
        rf"& {cells(first['arousal']['lookup'], 'arousal', False)} \\",
        *rows_with_groups(rows, 10),
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\tabledesc{Each row is a ContextPose model given the ticked input streams. The reference rows are "
        r"the runs of the pose-free stream ablation (\code{tab:eavf} in eclip\_streams.tex); the other rows add "
        r"one or both video streams.",
        r"\emph{Video streams} (\code{asd\_contextpose.video\_features}): at each of the 32 steps of the "
        r"pre-event, event and post-event windows, a 16-frame window at 15 fps (about 1 s) centred on the step "
        r"is cropped to the child (a square of 4.5 shoulder widths around the tracked shoulder midpoint, "
        r"interpolated across untracked frames; the full frame when a clip has no track).",
        r"\begin{itemize}",
        r"  \item VST: a frozen Kinetics-400 Video Swin Transformer (torchvision Swin3D-B, ImageNet-22K and Kinetics-400 weights); its last-stage tokens averaged over space and time (1024-d).",
        r"  \item H: hand-designed HOG-HOF descriptors after Kone\v{c}n\'{y} and Hagara (2013): gradient-orientation histograms of the window's middle frame and optical-flow-orientation histograms over the window (4$\times$4 cells), with the crop's median flow (camera motion and whole-body drift) removed, plus two log motion energies (274-d).",
        r"\end{itemize}",
        r"Each video stream enters with a child-tracked flag, gets its own temporal encoder and modality embedding, and its tokens join those the context query attends over (\code{--stream}). Stream dropout hides each stream of a training sample with probability 0.3.",
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
    rows = [(row, cells(r["model"], True)) for row, r in reg]
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{\textbf{Video streams as regression of the final valence and arousal annotation} "
        r"(cross-validated, 5 seeds). Streams and row groups as in Table~\ref{tab:video}. Higher Spearman, CCC, AP, MCC "
        r"and mAP and lower MAE are better.}",
        r"\label{tab:video_reg}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{cccccclccccccccccc}",
        r"\toprule",
        rf"\multicolumn{{{span}}}{{l}}{{}} & \multicolumn{{5}}{{c}}{{Valence}} & \multicolumn{{5}}{{c}}{{Arousal}} & \\",
        rf"\cmidrule(lr){{{span + 1}-{span + 5}}}\cmidrule(lr){{{span + 6}-{span + 10}}}",
        " & ".join(HEADERS) + r" & Model & Spearman & CCC & MAE & AP & MCC & Spearman & CCC & MAE & AP & MCC & mAP \\",
        r"\midrule",
        rf"\multicolumn{{{span}}}{{l}}{{Overall median}} & {cells(first['median'], False)} \\",
        rf" & $\checkmark$ & & & & & Annotation lookup (no network) & {cells(first['lookup'], False)} \\",
        *rows_with_groups(rows, 11),
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\tabledesc{Each row is the model of the same row in Table~\ref{tab:video}, trained to regress the "
        r"final annotation's score ($[-1, 1]$ scale) instead of classifying its sign; "
        r"predictions are averaged over 5 seeds. Reference rows are the runs of \code{tab:eavf\_reg} in "
        rf"eclip\_streams.tex. {first['n']} development clips; protocol, offsets and metrics as in "
        r"\code{tab:eavf\_reg} (AP ranks the clips with a positive final annotation above the rest; MCC "
        r"compares the signs of prediction and label). Bold marks the best model in each column.}",
        r"\end{table}",
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--output", type=Path, default=ROOT / "reports/video_streams.tex")
    parser.add_argument("--json", type=Path, help="also write the metrics as JSON")
    args = parser.parse_args(argv)

    calls, reg = [], []
    for row in ROWS:
        vot_run, reg_run = row[3]
        calls.append((row, binary(load(vot_run))))
        reg.append((row, regression(load(reg_run))))
        v, a = calls[-1][1]["valence"]["model"], calls[-1][1]["arousal"]["model"]
        r = reg[-1][1]["model"]
        print(
            f"{label(row[1], row[2]):42s} bal V {v['balanced_accuracy']:.3f} A {a['balanced_accuracy']:.3f}"
            f" AUC V {v['auc']:.2f} A {a['auc']:.2f} | rho V {r['valence']['spearman']:.2f}"
            f" A {r['arousal']['spearman']:.2f} mAP {r['map']:.3f}"
        )  # fmt: skip
    people = annotators(RUNS["A"][0])
    body = binary_table(calls, people) + "\n" + regression_table(reg)
    text = document(body)
    text = text.replace("Stream Ablation without Pose", "Video Streams")
    text = text.replace("eclip_stream_tables`", "video_stream_tables`")
    args.output.write_text(text)
    print(f"wrote {args.output}")
    if args.json:
        payload = {
            "calls": {label(r[1], r[2]): c for r, c in calls},
            "regression": {label(r[1], r[2]): g for r, g in reg},
            "annotators": people,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
