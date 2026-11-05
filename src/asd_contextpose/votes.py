"""Annotator votes: soft targets, majority classes and the human ceiling.

Each clip was rated by three blind annotators. Valence votes are positive /
neutral / negative and arousal votes are high / low only; "unknown" votes are
missing. The consensus labels average the votes weighted by confidence, which
splits identical vote patterns across class boundaries, so the classification
targets here come from the votes themselves:

- valence: positive vs not positive (neutral or negative)
- arousal: high vs low

The soft target is the share of known votes for the positive/high class (each
annotator counts once); the class is the annotators' majority, or None on a tie.
"""

from collections import defaultdict

import numpy as np

TARGETS = ("valence", "arousal")
# Workbook cell values -> vote; 2 (unknown) is missing.
VOTE_VALUES = {
    "valence": {"1": 1, "0.5": 0, "0": -1},
    "arousal": {"1": 1, "0": -1},
}
POSITIVE = {"valence": 1, "arousal": 1}  # the class each binary target detects


def annotators(record: dict) -> list[str]:
    return sorted(k.split("__", 1)[1] for k in record if k.startswith("valence__"))


def parse_votes(record: dict, target: str) -> list[int | None]:
    """Per-annotator votes from `label:confidence` cells; None when unknown."""
    votes = []
    for name in annotators(record):
        cell = record[f"{target}__{name}"]
        value = str(cell).split(":", 1)[0].strip() if cell is not None else ""
        votes.append(VOTE_VALUES[target].get(value))
    return votes


def encode(votes: list[int | None]) -> str:
    """Manifest form, e.g. "1|0|" (empty = unknown)."""
    return "|".join("" if v is None else str(v) for v in votes)


def decode(text: str) -> list[int | None]:
    return [None if v == "" else int(v) for v in text.split("|")] if text else []


def soft_target(votes: list[int | None], target: str) -> float:
    """Share of known votes for the positive (valence) / high (arousal) class."""
    known = [v for v in votes if v is not None]
    if not known:
        raise ValueError(f"no known {target} votes")
    return sum(v == POSITIVE[target] for v in known) / len(known)


def majority(votes: list[int | None], target: str) -> int | None:
    """1 if most known votes are positive/high, 0 if most are not, None on a tie."""
    known = [v == POSITIVE[target] for v in votes if v is not None]
    hits = sum(known)
    if not known or hits * 2 == len(known):
        return None
    return int(hits * 2 > len(known))


def binary_scores(labels: np.ndarray, decisions: np.ndarray) -> dict:
    """Accuracy and balanced accuracy of 0/1 decisions."""
    recalls = [
        (decisions[labels == k] == k).mean() for k in (0, 1) if (labels == k).any()
    ]
    return {
        "accuracy": float((decisions == labels).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
    }


def annotator_ceiling(all_votes: list[list[int | None]], target: str) -> dict:
    """Each annotator against the other two, where those two agree.

    The reference is the class the other two share; one annotator's binary call
    is scored against it. Pooled over annotators.
    """
    labels, decisions = [], []
    for votes in all_votes:
        for i, own in enumerate(votes):
            others = [v for j, v in enumerate(votes) if j != i]
            if own is None or None in others:
                continue
            calls = [v == POSITIVE[target] for v in others]
            if calls[0] == calls[1]:
                labels.append(int(calls[0]))
                decisions.append(int(own == POSITIVE[target]))
    if not labels:
        return {"n": 0}
    return {"n": len(labels), **binary_scores(np.array(labels), np.array(decisions))}


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Area under the ROC curve (rank form; ties count half). None if one class."""
    labels = np.asarray(labels).astype(bool)
    if labels.all() or not labels.any():
        return None
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    start = 0
    for end in range(1, len(scores) + 1):
        if end == len(scores) or scores[order[end]] != scores[order[start]]:
            ranks[order[start:end]] = (start + end + 1) / 2  # mean 1-based rank
            start = end
    positives = int(labels.sum())
    negatives = len(labels) - positives
    rank_sum = ranks[labels].sum() - positives * (positives + 1) / 2
    return float(rank_sum / (positives * negatives))


def vote_labels(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """[N, 2] vote shares and majority classes (-1 = tie) from manifest votes."""
    shares, classes = [], []
    for row in rows:
        votes = {t: decode(row[f"{t}_votes"]) for t in TARGETS}
        shares.append([soft_target(votes[t], t) for t in TARGETS])
        winners = [majority(votes[t], t) for t in TARGETS]
        classes.append([-1 if m is None else m for m in winners])
    return np.array(shares), np.array(classes)


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximising balanced accuracy of `score > threshold` (lowest on ties)."""
    values = np.unique(scores)
    candidates = np.r_[values[0] - 1, (values[:-1] + values[1:]) / 2]
    balanced = [
        binary_scores(labels, (scores > c).astype(int))["balanced_accuracy"]
        for c in candidates
    ]
    return float(candidates[int(np.argmax(balanced))])


def share_lookup(rows: list[dict], shares: np.ndarray, column: str) -> dict:
    """Mean vote share per value of `column` (groups under 3 use the overall)."""
    groups = defaultdict(list)
    for row, share in zip(rows, shares):
        groups[row.get(column, "")].append(share)
    return {
        "column": column,
        "default": shares.mean(0).tolist(),
        "groups": {k: np.mean(v, 0).tolist() for k, v in groups.items() if len(v) >= 3},
    }


def bootstrap_interval(groups, statistic, draws=2000, seed=0) -> list[float] | None:
    """95% interval of `statistic(indices)`, resampling participants.

    Resamples where the statistic is undefined (None) are skipped; None if all are.
    """
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in ids]
    values = []
    for _ in range(draws):
        picked = np.concatenate(
            [members[i] for i in rng.integers(len(ids), size=len(ids))]
        )
        value = statistic(picked)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return [float(q) for q in np.percentile(values, [2.5, 97.5])]


def classification_report(pooled: list[dict], all_rows: list[dict]) -> dict:
    """Binary calls vs the annotators' majority: model, lookup, majority, people."""
    groups = np.array([p["participant_id"] for p in pooled])
    report = {}
    for index, target in enumerate(TARGETS):
        labels = np.array([p["classes"][index] for p in pooled])
        keep = labels >= 0
        y, g = labels[keep], groups[keep]
        callers = {}
        for name in ("model", "lookup"):
            scores = np.array([p["scores"][name][index] for p in pooled])[keep]
            calls = np.array([p["calls"][name][index] for p in pooled])[keep]
            callers[name] = (scores, calls)
        majority = np.array([p["calls"]["majority"][index] for p in pooled])[keep]
        callers["majority"] = (np.zeros(len(y)), majority)

        def scores_of(name, idx):
            scores, calls = callers[name]
            return {
                **binary_scores(y[idx], calls[idx]),
                "auc": auc(scores[idx], y[idx]),
            }

        everything = np.arange(len(y))
        entry = {"n": int(len(y)), "positives": int(y.sum())}
        for name in callers:
            entry[name] = scores_of(name, everything)
        model_scores, model_calls = callers["model"]
        entry["model"]["balanced_accuracy_95ci"] = bootstrap_interval(
            g, lambda i: scores_of("model", i)["balanced_accuracy"]
        )
        entry["model"]["auc_95ci"] = bootstrap_interval(
            g, lambda i: auc(model_scores[i], y[i])
        )
        entry["model_vs_lookup_balanced_accuracy_95ci"] = bootstrap_interval(
            g,
            lambda i: (
                scores_of("model", i)["balanced_accuracy"]
                - scores_of("lookup", i)["balanced_accuracy"]
            ),
        )
        entry["annotators"] = annotator_ceiling(
            [decode(r[f"{target}_votes"]) for r in all_rows], target
        )
        report[target] = entry
    return report


def has_votes(rows: list[dict]) -> bool:
    return bool(rows) and all(f"{t}_votes" in rows[0] for t in TARGETS)


def lookup_scores(table: dict, rows: list[dict]) -> np.ndarray:
    return np.array(
        [
            table["groups"].get(r.get(table["column"], ""), table["default"])
            for r in rows
        ]
    )


def fit_calls(rows: list[dict], scores: np.ndarray, column: str) -> dict:
    """Decision rules for the binary calls, fitted on the clips a model saw.

    Thresholds maximise balanced accuracy for the model's scores and for a
    lookup of the mean vote share per `column` value; the majority class is
    the fallback. Plain lists, so it can be stored in a checkpoint.
    """
    shares, classes = vote_labels(rows)
    table = share_lookup(rows, shares, column)
    lookup = lookup_scores(table, rows)
    fit = {"lookup": table, "model_thresholds": [], "lookup_thresholds": []}
    fit["majority"] = []
    for index in range(len(TARGETS)):
        decided = classes[:, index] >= 0
        labels = classes[decided, index]
        fit["model_thresholds"].append(best_threshold(scores[decided, index], labels))
        fit["lookup_thresholds"].append(best_threshold(lookup[decided, index], labels))
        fit["majority"].append(int(labels.mean() > 0.5))
    return fit


def make_calls(fit: dict, rows: list[dict], scores: np.ndarray) -> dict:
    """Scores and 0/1 calls ([N, 2] each) for model, lookup and majority."""
    lookup = lookup_scores(fit["lookup"], rows)
    return {
        "scores": {"model": scores, "lookup": lookup},
        "calls": {
            "model": (scores > np.array(fit["model_thresholds"])).astype(int),
            "lookup": (lookup > np.array(fit["lookup_thresholds"])).astype(int),
            "majority": np.tile(fit["majority"], (len(rows), 1)),
        },
    }


def ensemble_report(
    rows: list[dict], preds: list[np.ndarray], fits: list[dict]
) -> dict:
    """`classification_report` for one model or an ensemble on labelled clips.

    Each member applies its own stored thresholds; the ensemble's call is the
    members' majority (ties count as negative) and its AUC uses the mean score.
    The lookup and majority baselines come from the first member.
    """
    made = [make_calls(fit, rows, pred) for fit, pred in zip(fits, preds)]
    calls = (np.mean([m["calls"]["model"] for m in made], 0) > 0.5).astype(int)
    scores = np.mean(preds, 0)
    _, classes = vote_labels(rows)
    entries = [
        {
            "participant_id": row["participant_id"],
            "classes": classes[i].tolist(),
            "scores": {
                "model": scores[i].tolist(),
                "lookup": made[0]["scores"]["lookup"][i].tolist(),
            },
            "calls": {
                "model": calls[i].tolist(),
                "lookup": made[0]["calls"]["lookup"][i].tolist(),
                "majority": made[0]["calls"]["majority"][i].tolist(),
            },
        }
        for i, row in enumerate(rows)
    ]
    return classification_report(entries, rows)
