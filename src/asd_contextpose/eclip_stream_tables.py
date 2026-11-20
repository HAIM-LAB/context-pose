"""Stream ablation without pose: EmotionCLIP (E), annotation (A), VLM (V), face (F).

Writes two tables over every combination of the four streams (singles, pairs,
triples, all four), from the 5-seed cross-validation runs already on disk:

* binary calls (votes-objective runs): accuracy, balanced accuracy with a 95%
  participant-bootstrap interval, F1 of the positive class, MCC and AUC, for
  valence positive vs not and arousal high vs low (annotator majority, ties
  excluded). The ensemble's call is the majority of the seeds' calls and its
  AUC uses the mean seed score, as in Table 1 (tab:streams).
* regression (regression-objective runs, consensus labels on [-1, 1]):
  Spearman rho, CCC, MAE, average precision and MCC per target, and mAP. The
  ensemble's prediction is the mean over seeds. AP ranks the clips with a
  positive consensus label above the rest; MCC compares the signs of the
  prediction and of the label (> 0 vs <= 0).

Rows without E reuse the runs of tab:streams and tab:streams_reg; rows with E
are the `--subject eclip` runs of tab:subject and tab:subject_reg.

    uv run python -m asd_contextpose.eclip_stream_tables \
        --output reports/eclip_streams.tex
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from asd_contextpose import votes
from asd_contextpose.regression_table import average_precision, ccc

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints"
STREAMS = "EAVF"
NAMES = {"E": "EmotionCLIP", "A": "annotation", "V": "VLM", "F": "face"}
# Run folders under checkpoints/ for each stream set: (votes, regression).
RUNS = {
    "E": ("subject/vot/E_subject", "subject/reg/E_subject"),
    "A": ("streams/T_A_context", "streams_reg/T_A_context"),
    "V": ("streams/V_vlm", "streams_reg/V_vlm"),
    "F": ("votes/V5_face_only", "streams_reg/F_face_only"),
    "EA": ("subject/vot/EA_subject_ann", "subject/reg/EA_subject_ann"),
    "EV": ("subject/vot/EV_subject_vlm", "subject/reg/EV_subject_vlm"),
    "EF": ("subject/vot/EF_subject_face", "subject/reg/EF_subject_face"),
    "AV": ("v3vlm/L0_vlm", "streams_reg/AV_ann_vlm"),
    "AF": ("votes/V3_context_face", "streams_reg/AF_context_face"),
    "VF": ("streams/VF_vlm_face", "streams_reg/VF_vlm_face"),
    "EAV": ("subject/vot/EAV_subject_ann_vlm", "subject/reg/EAV_subject_ann_vlm"),
    "EAF": ("subject/vot/EAF_subject_ann_face", "subject/reg/EAF_subject_ann_face"),
    "EVF": ("subject/vot/EVF_subject_vlm_face", "subject/reg/EVF_subject_vlm_face"),
    "AVF": ("streams/AVF_ann_vlm_face", "streams_reg/AVF_ann_vlm_face"),
    "EAVF": ("subject/vot/EAVF_all_four", "subject/reg/EAVF_all_four"),
}
TARGETS = votes.TARGETS


def load(run: str) -> list[dict]:
    """Each seed's predictions keyed by clip path, checked to cover the same clips."""
    files = sorted((CHECKPOINTS / run).glob("seed*/cv_report.json"))
    if len(files) != 5:
        print(f"note: {run} has {len(files)} seeds, not 5", file=sys.stderr)
    seeds = [{Path(p["path"]).name: p for p in json.load(open(f))["predictions"]} for f in files]  # fmt: skip
    assert all(s.keys() == seeds[0].keys() for s in seeds), f"{run}: seeds differ"
    return seeds


def mcc(labels: np.ndarray, calls: np.ndarray) -> float:
    """Matthews correlation of 0/1 calls (0 when a row or column of the table is empty)."""
    tp = float(np.sum(calls & labels))
    tn = float(np.sum(~calls & ~labels))
    fp = float(np.sum(calls & ~labels))
    fn = float(np.sum(~calls & labels))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else 0.0


def f1(labels: np.ndarray, calls: np.ndarray) -> float:
    tp = np.sum(calls & labels)
    return float(2 * tp / (calls.sum() + labels.sum())) if calls.any() or labels.any() else 0.0  # fmt: skip


def calls_metrics(labels, calls, scores, groups) -> dict:
    labels, calls = labels.astype(bool), calls.astype(bool)
    result = {
        **votes.binary_scores(labels.astype(int), calls.astype(int)),
        "f1": f1(labels, calls),
        "mcc": mcc(labels, calls),
        "auc": votes.auc(scores, labels) if scores is not None else None,
    }
    if groups is not None:
        result["balanced_accuracy_95ci"] = votes.bootstrap_interval(
            groups,
            lambda i: votes.binary_scores(labels[i].astype(int), calls[i].astype(int))["balanced_accuracy"],  # fmt: skip
        )
    return result


def binary(seeds: list[dict]) -> dict:
    """Ensemble metrics per target, plus the lookup and majority references."""
    paths = sorted(seeds[0])
    first = seeds[0]
    groups = np.array([first[p]["participant_id"] for p in paths])
    out = {}
    for index, target in enumerate(TARGETS):
        labels = np.array([first[p]["classes"][index] for p in paths])
        keep = labels >= 0
        y, g = labels[keep], groups[keep]

        def column(source, name, member=first):
            return np.array([member[p][source][name][index] for p in paths])[keep]

        model_calls = np.mean([column("calls", "model", s) for s in seeds], 0) > 0.5
        model_scores = np.mean([column("scores", "model", s) for s in seeds], 0)
        out[target] = {
            "n": int(keep.sum()),
            "positives": int(y.sum()),
            "model": calls_metrics(y, model_calls, model_scores, g),
            "lookup": calls_metrics(y, column("calls", "lookup"), column("scores", "lookup"), None),  # fmt: skip
            "majority": {**calls_metrics(y, column("calls", "majority"), None, None), "auc": 0.5},  # fmt: skip
        }
    return out


def spearman(pred: np.ndarray, target: np.ndarray) -> float | None:
    def ranks(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        for value in np.unique(x):  # ties share their mean rank
            r[x == value] = r[x == value].mean()
        return r

    a, b = ranks(pred), ranks(target)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    result = {}
    for index, name in enumerate(TARGETS):
        p, y = pred[:, index], target[:, index]
        positive = y > 0
        result[name] = {
            "spearman": spearman(p, y),
            "ccc": ccc(p, y),
            "mae": float(np.abs(p - y).mean()),
            "ap": average_precision(p, positive),
            "mcc": mcc(positive, p > 0),
        }
    result["map"] = float(np.mean([result[t]["ap"] for t in TARGETS]))
    return result


def regression(seeds: list[dict]) -> dict:
    paths = sorted(seeds[0])
    first = seeds[0]
    target = np.array([first[p]["target"] for p in paths], dtype=np.float64)
    pred = np.mean([[s[p]["pred"] for p in paths] for s in seeds], 0)
    baselines = {
        name: np.array([first[p]["baselines"][name] for p in paths], dtype=np.float64)
        for name in ("train_median", "event_text_median")
    }
    return {
        "model": regression_metrics(pred, target),
        "median": regression_metrics(baselines["train_median"], target),
        "lookup": regression_metrics(baselines["event_text_median"], target),
        "n": len(paths),
    }


def annotators(run: str) -> dict:
    """Each annotator vs the other two, as crossval stored it for the run's clips."""
    report = json.load(open(sorted((CHECKPOINTS / run).glob("seed*/cv_report.json"))[0]))
    return {t: report["classification"][t]["annotators"] for t in TARGETS}


# ---------------------------------------------------------------------------
# LaTeX


def ticks(streams: str) -> str:
    return " & ".join(r"$\checkmark$" if s in streams else "" for s in STREAMS)


def label(streams: str) -> str:
    if len(streams) == 4:
        return "All four streams"
    text = " + ".join(NAMES[s] for s in streams)
    return text[0].upper() + text[1:] + (" only" if len(streams) == 1 else "")


def number(value: float | None, digits: int = 2, best: float | None = None) -> str:
    if value is None:
        return "--"
    text = f"{round(value, digits) + 0.0:.{digits}f}"
    if text.startswith("-"):
        text = "$-$" + text[1:]
    return rf"\best{{{text}}}" if best is not None and round(value, digits) == round(best, digits) else text  # fmt: skip


def percent(value: float, best: float | None = None) -> str:
    text = f"{round(100 * value):.0f}\\%"
    return rf"\best{{{text}}}" if best is not None and round(100 * value) == round(100 * best) else text  # fmt: skip


def binary_table(results: dict[str, dict], people: dict) -> str:
    keys = ("accuracy", "balanced_accuracy", "f1", "mcc", "auc")
    best = {
        (t, k): max(r[t]["model"][k] for r in results.values() if r[t]["model"][k] is not None)  # fmt: skip
        for t in TARGETS
        for k in keys
    }
    any_run = next(iter(results.values()))

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

    n = {t: any_run[t]["n"] for t in TARGETS}
    pos = {t: any_run[t]["positives"] for t in TARGETS}
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{\textbf{Stream ablation without pose on the binary valence and arousal calls} "
        r"(cross-validated, 5-seed ensembles). E = EmotionCLIP, A = event annotation, V = VLM scene "
        r"description, F = face.}",
        r"\label{tab:eavf}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{cccclcccccccccc}",
        r"\toprule",
        rf" & & & & & \multicolumn{{5}}{{c}}{{Valence: positive vs not ($n{{=}}{n['valence']}$)}} "
        rf"& \multicolumn{{5}}{{c}}{{Arousal: high vs low ($n{{=}}{n['arousal']}$)}} \\",
        r"\cmidrule(lr){6-10}\cmidrule(lr){11-15}",
        r"E & A & V & F & Model & Acc. & Bal. acc. & F1 & MCC & AUC & Acc. & Bal. acc. & F1 & MCC & AUC \\",
        r"\midrule",
        rf"\multicolumn{{5}}{{l}}{{Annotators (each vs the other two)}} & {person('valence')} & {person('arousal')} \\",
        rf"\multicolumn{{5}}{{l}}{{Always majority}} & {cells(any_run['valence']['majority'], 'valence', False)} "
        rf"& {cells(any_run['arousal']['majority'], 'arousal', False)} \\",
        rf" & $\checkmark$ & & & Annotation lookup (no network) & {cells(any_run['valence']['lookup'], 'valence', False)} "
        rf"& {cells(any_run['arousal']['lookup'], 'arousal', False)} \\",
    ]  # fmt: skip
    previous = None
    for streams, r in results.items():
        if len(streams) != previous:
            lines.append(r"\midrule")
            previous = len(streams)
        lines.append(
            f"{ticks(streams)} & {label(streams)} & {cells(r['valence']['model'], 'valence', True)} "
            f"& {cells(r['arousal']['model'], 'arousal', True)} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\tabledesc{Each row is a ContextPose model given the ticked input streams, used alone, "
        r"in pairs, in threes and all together; pose is never an input. EmotionCLIP enters as the "
        r"subject stream in place of pose (frozen 512-d EmotionCLIP frame embeddings at each of the 32 "
        r"steps of the pre-event, event and post-event windows, with a child-tracked flag).",
        r"\emph{Streams:}",
        r"\begin{itemize}",
        r"  \item E: frozen EmotionCLIP frame embeddings, attended over by the context query as pose is in the stream-ablation table (\code{tab:streams} in test.tex).",
        r"  \item A: the four binary event indicators coded by annotators.",
        r"  \item V: a 384-d MiniLM embedding of Qwen3-Omni scene descriptions.",
        r"  \item F: frozen FER ViT face embeddings, kept only on steps that pass the frontal, YuNet-detector and brightness quality gate.",
        r"\end{itemize}",
        r"\emph{Offsets:} rows with A start from the event-annotation lookup's log-odds and learn a correction to it; rows without A start from a constant offset fitted on the training folds, so no annotation information reaches them.",
        r"\emph{Protocol:} as in the stream-ablation table (\code{tab:streams} in test.tex): 153 development clips, video-grouped 5-fold cross-validation with duplicate videos merged, votes objective, identical training settings for every row (width 64, dropout 0.3, weight decay 0.05, learning rate $3\times10^{-4}$, up to 60 epochs with patience 10). Rows without E are the runs of the stream-ablation table (\code{tab:streams} in test.tex).",
        rf"\emph{{Targets:}} annotator-majority calls with ties excluded: valence positive vs not positive ({n['valence']} clips, {pos['valence']} positive); arousal high vs low ({n['arousal']} clips, {pos['arousal']} high).",
        r"\emph{Metrics:}",
        r"\begin{itemize}",
        r"  \item Calls are the majority of the 5 seeds' calls; AUC uses the mean seed score. Decision thresholds are set on each model's own training clips, never on the scored fold.",
        r"  \item Balanced accuracy is the mean per-class recall (chance 50\%), with 95\% participant-bootstrap intervals in brackets.",
        r"  \item F1 is for the positive class (positive valence; high arousal). MCC is the Matthews correlation coefficient between calls and labels ($-1$ to 1, 0 for chance or a constant call).",
        r"\end{itemize}",
        r"\emph{References:} each annotator against the other two where those agree, always predicting the majority class, and the non-learned lookup (training-fold mean vote share per indicator combination). Bold marks the best model in each column.}",
        r"\end{table}",
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def regression_table(results: dict[str, dict]) -> str:
    keys = (("spearman", max), ("ccc", max), ("mae", min), ("ap", max), ("mcc", max))
    best = {}
    for t in TARGETS:
        for k, pick in keys:
            best[(t, k)] = pick(r["model"][t][k] for r in results.values() if r["model"][t][k] is not None)  # fmt: skip
    best["map"] = max(r["model"]["map"] for r in results.values())
    any_run = next(iter(results.values()))

    def cells(m: dict, is_model: bool) -> str:
        parts = []
        for t in TARGETS:
            for k, _ in keys:
                digits = 3 if k == "mae" else 2
                parts.append(number(m[t][k], digits, best[(t, k)] if is_model else None))
        parts.append(number(m["map"], 3, best["map"] if is_model else None))
        return " & ".join(parts)

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{\textbf{Stream ablation without pose as regression of the consensus valence and arousal labels} "
        r"(cross-validated, 5 seeds). E = EmotionCLIP, A = event annotation, V = VLM scene description, F = face. "
        r"Higher Spearman, CCC, AP, MCC and mAP and lower MAE are better.}",
        r"\label{tab:eavf_reg}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{cccclccccccccccc}",
        r"\toprule",
        r" & & & & & \multicolumn{5}{c}{Valence} & \multicolumn{5}{c}{Arousal} & \\",
        r"\cmidrule(lr){6-10}\cmidrule(lr){11-15}",
        r"E & A & V & F & Model & Spearman & CCC & MAE & AP & MCC & Spearman & CCC & MAE & AP & MCC & mAP \\",
        r"\midrule",
        rf"\multicolumn{{5}}{{l}}{{Overall median}} & {cells(any_run['median'], False)} \\",
        rf" & $\checkmark$ & & & Annotation lookup (no network) & {cells(any_run['lookup'], False)} \\",
    ]  # fmt: skip
    previous = None
    for streams, r in results.items():
        if len(streams) != previous:
            lines.append(r"\midrule")
            previous = len(streams)
        lines.append(f"{ticks(streams)} & {label(streams)} & {cells(r['model'], True)} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\tabledesc{Each row is the model of the same row in Table~\ref{tab:eavf}, trained to regress the confidence-weighted consensus labels ($[-1, 1]$ scale) instead of the annotators' vote shares.",
        rf"\emph{{Protocol:}} as in the stream-ablation regression table (\code{{tab:streams\_reg}} in test.tex): {any_run['n']} development clips, video-grouped 5-fold cross-validation with duplicate videos merged, squared error on the consensus labels, identical training settings for every row; predictions are averaged over 5 seeds. Rows without E are the runs of the stream-ablation regression table (\code{{tab:streams\_reg}} in test.tex).",
        r"\emph{Offsets:} rows with A start from the training folds' median label per indicator combination (the lookup) and learn a correction; rows without A start from the training folds' overall median, so no annotation information reaches them.",
        r"\emph{Metrics:}",
        r"\begin{itemize}",
        r"  \item Spearman is the rank correlation between prediction and label; CCC is the concordance correlation coefficient, which also penalises bias and a compressed scale (0 for a constant prediction).",
        r"  \item MAE is the mean absolute error on the $[-1, 1]$ scale.",
        r"  \item AP is the average precision with which the prediction ranks the clips with a positive consensus label (positive valence; high arousal) above the rest; mAP is its mean over valence and arousal.",
        r"  \item MCC is the Matthews correlation coefficient between the signs of the prediction and of the label ($>0$ vs $\le 0$); it is 0 when every prediction falls on one side of 0, as it does for most valence rows (80\% of the labels are positive).",
        r"\end{itemize}",
        r"\emph{References:} the training folds' overall median and the lookup are non-learned. The overall median is constant within a fold, so its Spearman and AP reflect only the differences between folds' medians. Bold marks the best model in each column.}",
        r"\end{table}",
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def document(body: str) -> str:
    return "\n".join([
        r"\documentclass[11pt]{article}",
        "",
        r"\usepackage[margin=2.3cm]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{caption}",
        r"\usepackage[colorlinks=true,linkcolor=blue!50!black]{hyperref}",
        "",
        r"\captionsetup{font=small,labelfont=bf,justification=justified,singlelinecheck=false}",
        r"\newcommand{\code}[1]{\texttt{#1}}",
        r"\newcommand{\best}[1]{\textbf{#1}}",
        r"\newcommand{\ci}[2]{\,[#1,\,#2]}",
        r"% Description paragraph set below a table, inside its float.",
        r"\newcommand{\tabledesc}[1]{\par\medskip\begin{minipage}{\linewidth}\footnotesize\textbf{Description.} #1\end{minipage}}",
        "",
        r"\title{ASD-ContextPose: Stream Ablation without Pose}",
        r"\author{ASD-ContextPose project}",
        r"\date{\today}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"% Generated by `uv run python -m asd_contextpose.eclip_stream_tables`.",
        r"% Companion of tab:streams and tab:streams_reg in test.tex.",
        "",
        body,
        r"\end{document}",
        "",
    ])  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--output", type=Path, default=ROOT / "reports/eclip_streams.tex")
    parser.add_argument("--json", type=Path, help="also write the metrics as JSON")
    args = parser.parse_args(argv)

    calls, reg = {}, {}
    for streams, (vot_run, reg_run) in RUNS.items():
        calls[streams] = binary(load(vot_run))
        reg[streams] = regression(load(reg_run))
        v, a = calls[streams]["valence"]["model"], calls[streams]["arousal"]["model"]
        print(f"{streams:5s} calls V bal {v['balanced_accuracy']:.3f} A bal {a['balanced_accuracy']:.3f}"
              f" | reg mAP {reg[streams]['model']['map']:.3f}")  # fmt: skip
    people = annotators(RUNS["A"][0])
    body = binary_table(calls, people) + "\n" + regression_table(reg)
    args.output.write_text(document(body))
    print(f"wrote {args.output}")
    if args.json:
        args.json.write_text(json.dumps({"calls": calls, "regression": reg, "annotators": people}, indent=2) + "\n")  # fmt: skip


if __name__ == "__main__":
    main()
