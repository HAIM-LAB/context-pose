"""Participant-grouped k-fold cross-validation over development manifests.

Each fold is held out once for evaluation. The next fold is the validation set
used for early stopping, and the remaining folds train, so no clip is scored by
a checkpoint selected on it. Out-of-fold predictions are pooled and compared
with baselines that predict the training median overall and per value of
`--baseline-by` (the event text by default); a participant-level bootstrap
gives an interval for each difference. Pass only train/val manifests;
the test set stays untouched. Unrecognised options are forwarded to `train`.

When the manifests carry annotator votes (`prepare`), the report also scores
two binary calls against the annotators' majority: positive vs not-positive
valence and high vs low arousal (ties left out). The model's output is the
score; its decision threshold is chosen per fold on the clips the model was
fitted and selected on, maximising balanced accuracy. It is compared with
always predicting the majority class, a lookup of the mean vote share per
`--baseline-by` value (thresholded the same way), and the annotators
themselves (each against the other two).
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import train as training
from . import votes as annotator_votes
from .data import PoseDataset
from .model import ModelConfig, build_model

VOTE_TARGETS = annotator_votes.TARGETS
classification_report = annotator_votes.classification_report


def assign_folds(rows: list[dict], folds: int, seed: int) -> dict[str, int]:
    """Fold per participant, dealing each category's participants round-robin."""
    by_category = defaultdict(set)
    for row in rows:
        by_category[row.get("category", "")].add(row["participant_id"])
    rng = random.Random(seed)
    fold_of, offset = {}, 0
    for category in sorted(by_category):
        participants = sorted(by_category[category] - fold_of.keys())
        rng.shuffle(participants)
        for index, participant in enumerate(participants):
            fold_of[participant] = (offset + index) % folds
        offset += len(participants)
    return fold_of


def read_rows(manifests: list[Path]) -> tuple[list[dict], list[str]]:
    rows, fields = [], []
    for manifest in manifests:
        with manifest.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields += [f for f in reader.fieldnames or [] if f not in fields]
            for row in reader:
                row["path"] = str((manifest.parent / row["path"]).resolve())
                rows.append(row)
    return rows, fields


def write_manifest(path: Path, rows: list[dict], split: str, fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows({**row, "split": split} for row in rows)


def predict(
    checkpoint: Path, manifest: Path, device: torch.device, split: str = "test"
) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**state["config"])
    model = build_model(config).to(device)
    model.load_state_dict(state["state_dict"])
    dataset = PoseDataset(manifest, config, split)
    loader = DataLoader(training.WithOffset(dataset, state.get("residual")), 64)
    out = training.predict(model, loader, device)
    return {
        **out,
        "rows": dataset.rows,
        "best_epoch": state["epoch"],
        "config": config,
    }


def labelled(manifest: Path, config: ModelConfig, split: str):
    dataset = PoseDataset(manifest, config, split)
    targets = np.stack([dataset[i]["target"].numpy() for i in range(len(dataset))])
    return dataset.rows, targets


def bootstrap_gain(groups, model_err, base_err, draws=2000, seed=0):
    """95% interval of baseline minus model mean MAE, resampling participants."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in ids]
    gains = []
    for _ in range(draws):
        picked = np.concatenate(
            [members[i] for i in rng.integers(len(ids), size=len(ids))]
        )
        gains.append(base_err[picked].mean() - model_err[picked].mean())
    return [float(q) for q in np.percentile(gains, [2.5, 97.5])]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/cv"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--baseline-by",
        default="event_text",
        metavar="COLUMN",
        help="also compare with the training median per value of this column",
    )
    args, train_args = parser.parse_known_args(argv)
    if args.folds < 3:
        parser.error("--folds must be at least 3 (evaluation, validation, training)")

    rows, fields = read_rows(args.manifests)
    if any(row["split"] == "test" for row in rows):
        parser.error("cross-validation must not use test manifests")
    fold_of = assign_folds(rows, args.folds, args.fold_seed)
    has_votes = all(f"{t}_votes" in fields for t in VOTE_TARGETS)
    device = torch.device(args.device)
    pooled, per_fold = [], []
    for fold in range(args.folds):
        inner = (fold + 1) % args.folds
        directory = args.output / f"fold_{fold}"
        parts = {
            "test": [r for r in rows if fold_of[r["participant_id"]] == fold],
            "val": [r for r in rows if fold_of[r["participant_id"]] == inner],
        }
        parts["train"] = [
            r for r in rows if fold_of[r["participant_id"]] not in (fold, inner)
        ]
        for split, part in parts.items():
            write_manifest(directory / f"{split}.csv", part, split, fields)
        print(f"fold {fold}: " + ", ".join(f"{s}={len(p)}" for s, p in parts.items()))
        training.main(
            [
                "--train-manifest", str(directory / "train.csv"),
                "--val-manifest", str(directory / "val.csv"),
                "--output", str(directory / "best.pt"),
                "--device", args.device,
                *train_args,
            ]
        )  # fmt: skip
        result = predict(directory / "best.pt", directory / "test.csv", device)
        votes_model = result["config"].objective == "votes"
        if has_votes:
            fitted = [
                predict(directory / "best.pt", directory / f"{s}.csv", device, s)
                for s in ("train", "val")
            ]
            calls_fit = annotator_votes.fit_calls(
                fitted[0]["rows"] + fitted[1]["rows"],
                np.concatenate([f["pred"] for f in fitted]),
                args.baseline_by,
            )
            made = annotator_votes.make_calls(calls_fit, result["rows"], result["pred"])
            test_shares, test_classes = annotator_votes.vote_labels(result["rows"])
        seen_rows, seen = [], []
        for split in ("train", "val"):
            split_rows, split_targets = labelled(
                directory / f"{split}.csv", result["config"], split
            )
            seen_rows += split_rows
            seen.append(split_targets)
        grouped = training.group_medians(
            seen_rows, np.concatenate(seen), args.baseline_by
        )
        if votes_model:
            metrics = training.vote_metrics(result["pred"], result["votes"])
        else:
            metrics = training.regression_metrics(result["pred"], result["target"])
        per_fold.append(
            {"fold": fold, "best_epoch": result["best_epoch"], "model": metrics}
        )
        for index, (row, pred, target) in enumerate(
            zip(result["rows"], result["pred"], result["target"])
        ):
            group = row.get(args.baseline_by, "")
            entry = {
                "fold": fold,
                "path": row["path"],
                "participant_id": row["participant_id"],
                "category": row.get("category", ""),
                "target": target.tolist(),
                "pred": pred.tolist(),
                "baselines": {
                    "train_median": grouped["default"],
                    f"{args.baseline_by}_median": grouped["groups"].get(
                        group, grouped["default"]
                    ),
                },
            }
            if has_votes:
                entry.update(
                    {
                        "votes": test_shares[index].tolist(),
                        "classes": test_classes[index].tolist(),
                        "scores": {
                            k: v[index].tolist() for k, v in made["scores"].items()
                        },
                        "calls": {
                            k: v[index].tolist() for k, v in made["calls"].items()
                        },
                    }
                )
            pooled.append(entry)

    pred = np.array([p["pred"] for p in pooled])
    target = np.array([p["target"] for p in pooled])
    groups = np.array([p["participant_id"] for p in pooled])
    model_err = np.abs(pred - target).mean(1)
    baselines = {}
    for name in pooled[0]["baselines"]:
        base = np.array([p["baselines"][name] for p in pooled])
        base_err = np.abs(base - target).mean(1)
        baselines[name] = {"metrics": training.regression_metrics(base, target)}
        if not votes_model:  # log-odds are not on the label scale
            baselines[name]["model_gain"] = float(base_err.mean() - model_err.mean())
            baselines[name]["model_gain_95ci"] = bootstrap_gain(
                groups, model_err, base_err
            )
    by_category = {}
    for category in sorted({p["category"] for p in pooled}):
        mask = np.array([p["category"] == category for p in pooled])
        by_category[category] = {
            "n": int(mask.sum()),
            "model_mean_mae": float(model_err[mask].mean()),
        }
    report = {
        "folds": args.folds,
        "fold_seed": args.fold_seed,
        "train_args": train_args,
        "objective": "votes" if votes_model else "regression",
        "model": (
            training.vote_metrics(pred, np.array([p["votes"] for p in pooled]))
            if votes_model
            else training.regression_metrics(pred, target)
        ),
        "baselines": baselines,
        "by_category": by_category,
        "per_fold": per_fold,
        "predictions": pooled,
    }
    if has_votes:
        report["classification"] = classification_report(pooled, rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cv_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    if votes_model:
        print(f"model mean cross-entropy {report['model']['mean_bce']:.4f}")
    else:
        print(f"model mean MAE {report['model']['mean_mae']:.4f}")
        for name, baseline in baselines.items():
            print(
                f"{name}: mean MAE {baseline['metrics']['mean_mae']:.4f}; model gain"
                f" {baseline['model_gain']:+.4f}, 95% CI {baseline['model_gain_95ci']}"
            )
    for target, entry in report.get("classification", {}).items():
        model, lookup = entry["model"], entry["lookup"]
        low, high = model["balanced_accuracy_95ci"]
        print(
            f"{target}: balanced accuracy model {model['balanced_accuracy']:.3f}"
            f" [{low:.3f}, {high:.3f}], lookup {lookup['balanced_accuracy']:.3f},"
            f" annotators {entry['annotators'].get('balanced_accuracy', float('nan')):.3f};"
            f" AUC model {model['auc'] if model['auc'] is None else round(model['auc'], 3)}"
        )


if __name__ == "__main__":
    main()
