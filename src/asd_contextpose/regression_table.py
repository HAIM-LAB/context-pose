"""Regression metrics for the stream-ablation runs (the companion of Table 1).

Reads the `cv_report.json` files that `crossval` wrote for votes-objective
runs, groups the seed repeats of each configuration, and scores each
ensemble's predicted vote share (sigmoid of the log-odds, averaged over seeds)
against the annotators' observed vote share (share of positive-valence /
high-arousal votes, in [0, 1]):

* MAE, RMSE and CCC for valence and arousal, and their means;
* mAP: the mean over the two targets of the average precision with which the
  predicted share ranks the clips whose annotator majority is positive
  valence / high arousal above the rest (ties left out, as in Table 1).

Reference rows: always predicting the mean share of the clips outside the
scored fold, and the event-annotation lookup (mean share per indicator
combination on the clips each model was fitted on, as stored in the report).

The streams of each run (P pose, A annotation, V VLM, F face) are read from
its training arguments and printed, so check the mapping before using the
table. Run it where the cross-validation outputs live:

    uv run python -m asd_contextpose.regression_table checkpoints/streams \
        --output reports/regression_table.tex
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TARGETS = ("valence", "arousal")
STREAMS = "PAVF"
NAMES = {"P": "pose", "A": "annotation", "V": "VLM", "F": "face"}
# Row order of Table 1: singles, pairs, triples, all four.
ORDER = [
    "P", "A", "V", "F",
    "PA", "PV", "PF", "AV", "AF", "VF",
    "PAF", "PVF", "PAV", "AVF",
    "PAVF",
]  # fmt: skip
SEED_FLAGS = {"--seed"}


def option(args: list[str], flag: str, default: str | None = None) -> str | None:
    """Value of `flag` in an argument list (`--flag value` or `--flag=value`)."""
    value = default
    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            value = args[index + 1]
        elif arg.startswith(flag + "="):
            value = arg.split("=", 1)[1]
    return value


def streams_of(args: list[str]) -> str:
    """P/A/V/F letters a run used, following the model's `--inputs` rules.

    Pose enters unless `--inputs` is text or face; the context vector enters
    for all/text and is the annotation (4-d, or an annotation_* model) or the
    VLM embedding (384-d); a lookup offset (`--residual-by`) brings in the
    annotation as well; the face stream is added whenever `--face-dim` > 0.
    """
    inputs = option(args, "--inputs", "all")
    model = option(args, "--model", "text_attention")
    text_dim = option(args, "--text-dim")
    used = set()
    if inputs in ("all", "pose"):
        used.add("P")
    if inputs in ("all", "text"):
        if model.startswith("annotation") or text_dim == "4":
            used.add("A")
        elif text_dim == "384":
            used.add("V")
        else:
            used.add("C")  # another context (e.g. CLIP text, 768-d)
    if option(args, "--residual-by"):
        used.add("A")
    if int(option(args, "--face-dim", "0")) > 0:
        used.add("F")
    return "".join(s for s in STREAMS + "C" if s in used)


def config_key(args: list[str]) -> tuple[str, ...]:
    """Training arguments without the seed, to group repeats of one setting."""
    kept, skip = [], False
    for arg in args:
        if skip:
            skip = False
        elif arg in SEED_FLAGS:
            skip = True
        elif not any(arg.startswith(f + "=") for f in SEED_FLAGS):
            kept.append(arg)
    return tuple(kept)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mean precision at each positive, ranking by score (ties share a rank)."""
    labels = np.asarray(labels).astype(bool)
    if not labels.any():
        return None
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    scores, labels = np.asarray(scores)[order], labels[order]
    precision_sum, hits, start = 0.0, 0, 0
    while start < len(scores):
        end = start
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        block = int(labels[start:end].sum())
        hits += block
        precision_sum += block * hits / end  # tied block counted at its end
        start = end
    return float(precision_sum / labels.sum())


def ccc(pred: np.ndarray, target: np.ndarray) -> float | None:
    denominator = pred.var() + target.var() + (pred.mean() - target.mean()) ** 2
    if len(target) < 2 or denominator <= 0:
        return None
    return float(2 * np.mean((pred - pred.mean()) * (target - target.mean())) / denominator)


def score(pred: np.ndarray, share: np.ndarray, classes: np.ndarray) -> dict:
    """Regression metrics on vote shares plus mAP on the majority classes."""
    result = {"n": len(share)}
    precisions = []
    for index, name in enumerate(TARGETS):
        p, y = pred[:, index], share[:, index]
        decided = classes[:, index] >= 0
        ap = average_precision(p[decided], classes[decided, index] == 1)
        precisions.append(ap)
        result[name] = {
            "mae": float(np.abs(p - y).mean()),
            "rmse": float(np.sqrt(np.square(p - y).mean())),
            "ccc": ccc(p, y),
            "ap": ap,
        }
    result["mean_mae"] = float(np.mean([result[t]["mae"] for t in TARGETS]))
    result["mean_rmse"] = float(np.mean([result[t]["rmse"] for t in TARGETS]))
    result["map"] = (
        float(np.mean(precisions)) if all(a is not None for a in precisions) else None
    )
    return result


def load(paths: list[Path]) -> dict[tuple, dict]:
    """Votes-objective reports under `paths`, grouped by configuration."""
    files = []
    for path in paths:
        files += [path] if path.is_file() else sorted(path.rglob("cv_report.json"))
    groups: dict[tuple, dict] = {}
    for file in files:
        report = json.loads(file.read_text())
        if report.get("objective") != "votes":
            print(f"skipped (not a votes run): {file}", file=sys.stderr)
            continue
        args = report.get("train_args", [])
        key = config_key(args)
        entry = groups.setdefault(
            key, {"streams": streams_of(args), "files": [], "reports": []}
        )
        entry["files"].append(file)
        entry["reports"].append(report)
    return groups


def ensemble(reports: list[dict]) -> dict:
    """Mean predicted share over seeds, aligned on clip path."""
    by_path = [{p["path"]: p for p in r["predictions"]} for r in reports]
    paths = sorted(set.intersection(*(set(b) for b in by_path)))
    if any(len(b) != len(paths) for b in by_path):
        print("warning: seeds scored different clips; using the shared ones",
              file=sys.stderr)  # fmt: skip
    first = by_path[0]
    pred = np.mean(
        [[sigmoid(np.array(b[p]["pred"])) for p in paths] for b in by_path], 0
    )
    lookup = np.mean(
        [[b[p]["scores"]["lookup"] for p in paths] for b in by_path], 0
    )
    return {
        "pred": pred,
        "lookup": lookup,
        "share": np.array([first[p]["votes"] for p in paths], dtype=np.float64),
        "classes": np.array([first[p]["classes"] for p in paths]),
        "fold": np.array([first[p]["fold"] for p in paths]),
    }


def constant_share(share: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Each clip gets the mean share of the clips outside its fold."""
    out = np.empty_like(share)
    for f in np.unique(fold):
        out[fold == f] = share[fold != f].mean(0)
    return out


def label(streams: str) -> str:
    words = [NAMES.get(s, "CLIP text") for s in streams]
    if len(words) == 4:
        return "All four streams"
    text = " + ".join(words)
    return text[0].upper() + text[1:] + (" only" if len(words) == 1 else "")


def fmt(value: float | None, best: float | None = None) -> str:
    if value is None:
        return "--"
    text = f"{round(value, 3) + 0.0:.3f}".replace("-", "$-$")  # + 0.0: no "-0.000"
    return rf"\best{{{text}}}" if best is not None and round(value, 3) == round(best, 3) else text


def latex(rows: list[tuple[str, str, dict]], n: int) -> str:
    """Table in the style of Table 2, with Table 1's stream columns."""
    columns = {
        "mean_mae": min, "mean_rmse": min, "map": max,
        ("valence", "mae"): min, ("valence", "rmse"): min, ("valence", "ccc"): max,
        ("arousal", "mae"): min, ("arousal", "rmse"): min, ("arousal", "ccc"): max,
    }  # fmt: skip

    def get(metrics, column):
        return metrics[column[0]][column[1]] if isinstance(column, tuple) else metrics[column]

    best = {}
    for column, pick in columns.items():
        values = [get(m, column) for kind, _, m in rows if kind == "model"]
        values = [v for v in values if v is not None]
        best[column] = pick(values) if values else None
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{\textbf{Stream ablation of ContextPose as regression of the annotators' vote shares} "
        rf"(the runs of Table~\ref{{tab:streams}}; {n} development clips, video-grouped 5-fold "
        r"cross-validation, 5-seed ensembles). Each model's prediction is the predicted vote share "
        r"$\sigma(\hat{y})$ for positive valence and for high arousal, averaged over seeds, compared "
        r"with the observed share of annotator votes ($[0,1]$ scale). "
        r"\emph{Metrics:} MAE and RMSE (lower is better); CCC = concordance correlation coefficient "
        r"(higher is better; 0 for a constant prediction); mean MAE and mean RMSE average valence (V) "
        r"and arousal (A); mAP = mean over V and A of the average precision for the annotators' majority "
        r"class (positive valence; high arousal; ties excluded). "
        r"\emph{References:} always predicting the mean share of the training folds, and the "
        r"event-annotation lookup (mean share per indicator combination on each model's training clips). "
        r"Bold: best model in each column.}",
        r"\label{tab:streams_regression}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{cccclccccc}",
        r"\toprule",
        r"P & A & V & F & Model & Mean MAE & Mean RMSE & mAP & V MAE / RMSE / CCC & A MAE / RMSE / CCC \\",
        r"\midrule",
    ]
    previous = None
    for kind, streams, metrics in rows:
        group = "ref" if kind != "model" else len(streams)
        if previous is not None and group != previous:
            lines.append(r"\midrule")
        previous = group
        ticks = " & ".join(r"$\checkmark$" if s in streams else "" for s in STREAMS)
        name = {"constant": "Training-fold mean share",
                "lookup": "Event-annotation lookup (no network)"}.get(kind) or label(streams)  # fmt: skip
        b = (lambda c: best[c]) if kind == "model" else (lambda c: None)
        per_target = [
            " / ".join(fmt(metrics[t][k], b((t, k))) for k in ("mae", "rmse", "ccc"))
            for t in TARGETS
        ]
        lines.append(
            f"{ticks} & {name} & {fmt(metrics['mean_mae'], b('mean_mae'))} & "
            f"{fmt(metrics['mean_rmse'], b('mean_rmse'))} & {fmt(metrics['map'], b('map'))} & "
            f"{per_target[0]} & {per_target[1]} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("paths", type=Path, nargs="+",
                        help="run folders (searched for cv_report.json) or report files")  # fmt: skip
    parser.add_argument("--output", type=Path, help="write the LaTeX table here")
    parser.add_argument("--json", type=Path, help="also write the metrics as JSON")
    args = parser.parse_args(argv)

    groups = load(args.paths)
    if not groups:
        parser.error("no votes-objective cv_report.json found")
    by_streams = defaultdict(list)
    for entry in groups.values():
        by_streams[entry["streams"]].append(entry)

    rows, reference, summary = [], None, {}
    for streams in ORDER + sorted(set(by_streams) - set(ORDER)):
        for entry in by_streams.get(streams, []):
            data = ensemble(entry["reports"])
            metrics = score(data["pred"], data["share"], data["classes"])
            folders = sorted({str(f.parent) for f in entry["files"]})
            where = os.path.commonpath(folders) if len(folders) > 1 else folders[0]
            print(f"{streams:5s} {label(streams):40s} seeds={len(entry['reports'])} "
                  f"mean MAE={metrics['mean_mae']:.3f}  <- {where}")  # fmt: skip
            if len(by_streams[streams]) > 1:
                print(f"      warning: {len(by_streams[streams])} settings share streams "
                      f"{streams}; pass only the folders you want", file=sys.stderr)  # fmt: skip
            if len(entry["reports"]) != 5:
                print(f"      note: {len(entry['reports'])} seeds, not 5", file=sys.stderr)
            rows.append(("model", streams, metrics))
            summary.setdefault(streams, []).append({"folders": folders, **metrics})
            if reference is None:
                reference = data
    constant = score(constant_share(reference["share"], reference["fold"]),
                     reference["share"], reference["classes"])  # fmt: skip
    lookup = score(reference["lookup"], reference["share"], reference["classes"])
    rows = [("constant", "", constant), ("lookup", "A", lookup)] + rows
    summary["references"] = {"training_fold_mean": constant, "lookup": lookup}

    table = latex(rows, len(reference["share"]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table)
        print(f"wrote {args.output}")
    else:
        print(table)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
