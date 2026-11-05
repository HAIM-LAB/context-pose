"""Small CPU-first training/evaluation CLI; no text or video processing."""

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import votes as annotator_votes
from .data import POSE_KEYS, PoseDataset, assert_disjoint, assert_group_separation
from .model import (
    CONTEXT_DIM,
    FACE_TARGETS,
    INPUTS,
    MODELS,
    OBJECTIVES,
    ContextPose,
    ModelConfig,
    build_model,
)
from .votes import auc

MIN_RATE = 0.02  # vote shares are clipped before taking log-odds


def group_medians(rows: list[dict], targets: np.ndarray, column: str) -> dict:
    """Median target per value of a manifest column (groups under 3 use the overall)."""
    groups = defaultdict(list)
    for row, target in zip(rows, targets):
        groups[row.get(column, "")].append(target)
    return {
        "column": column,
        "default": np.median(targets, 0).tolist(),
        "groups": {
            key: np.median(values, 0).tolist()
            for key, values in groups.items()
            if len(values) >= 3
        },
    }


def group_logits(rows: list[dict], votes: np.ndarray, column: str) -> dict:
    """Log-odds of the mean vote share per value of a manifest column.

    Same layout as `group_medians` (groups under 3 use the overall share), so a
    `votes` model can predict on top of it with `WithOffset`.
    """

    def logit(share: np.ndarray) -> list[float]:
        share = np.clip(share, MIN_RATE, 1 - MIN_RATE)
        return np.log(share / (1 - share)).tolist()

    groups = defaultdict(list)
    for row, vote in zip(rows, votes):
        groups[row.get(column, "")].append(vote)
    return {
        "column": column,
        "default": logit(votes.mean(0)),
        "groups": {
            key: logit(np.mean(values, 0))
            for key, values in groups.items()
            if len(values) >= 3
        },
    }


class WithOffset(Dataset):
    """Adds each sample's `offset`: the residual baseline the model predicts on top of."""

    def __init__(self, dataset: PoseDataset, residual: dict | None) -> None:
        self.dataset = dataset
        offsets = [
            residual["groups"].get(row.get(residual["column"], ""), residual["default"])
            if residual
            else [0.0, 0.0]
            for row in dataset.rows
        ]
        self.offsets = torch.tensor(offsets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        return {**self.dataset[index], "offset": self.offsets[index]}


def mirror_index(joint_order: str) -> list[int] | None:
    """Joint permutation swapping left_*/right_* names, or None if not pairable."""
    names = joint_order.split("|") if joint_order else []
    swapped = [
        n.replace("left", "\0").replace("right", "left").replace("\0", "right")
        for n in names
    ]
    if not names or set(swapped) != set(names):
        return None
    return [names.index(n) for n in swapped]


def augment(inputs: dict, mirror: list[int] | None, generator: torch.Generator) -> dict:
    """Per-sample rotation (±15°), scale (0.9-1.1), jitter and left/right mirror.

    The same transform applies to all three regions of a sample. Only observed
    joints (confidence > 0, when present) move; missing joints stay zero.
    """
    poses = torch.cat([inputs[key] for key in POSE_KEYS], dim=1)
    batch = poses.shape[0]
    xy, rest = poses[..., :2], poses[..., 2:]
    observed = (
        (rest[..., :1] > 0).float() if rest.shape[-1] else torch.ones_like(xy[..., :1])
    )
    angle = (torch.rand(batch, generator=generator) * 2 - 1) * math.radians(15)
    scale = 0.9 + 0.2 * torch.rand(batch, generator=generator)
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    rotation = torch.stack(
        [torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2
    )
    xy = torch.einsum("btjc,bdc->btjd", xy, rotation)
    xy = xy + 0.02 * torch.randn(xy.shape, generator=generator)
    poses = torch.cat([xy * observed, rest], dim=-1)
    if mirror is not None:
        flipped = poses[:, :, mirror] * torch.tensor(
            [-1.0, 1.0] + [1.0] * rest.shape[-1]
        )
        flip = torch.rand(batch, generator=generator) < 0.5
        poses = torch.where(flip[:, None, None, None], flipped, poses)
    steps = inputs[POSE_KEYS[0]].shape[1]
    return {**inputs, **dict(zip(POSE_KEYS, poses.split(steps, dim=1)))}


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pred, target = pred.astype(np.float64), target.astype(np.float64)
    result = {"n": len(target)}
    for index, name in enumerate(("valence", "arousal")):
        p, y = pred[:, index], target[:, index]
        denominator = p.var() + y.var() + (p.mean() - y.mean()) ** 2
        # CCC is undefined for singleton groups or identical constant series.
        ccc = (
            float(2 * np.mean((p - p.mean()) * (y - y.mean())) / denominator)
            if len(y) > 1 and denominator > 0
            else None
        )
        result[name] = {
            "mae": float(np.abs(p - y).mean()),
            "rmse": float(np.sqrt(np.square(p - y).mean())),
            "ccc": ccc,
        }
    result["mean_mae"] = (result["valence"]["mae"] + result["arousal"]["mae"]) / 2
    return result


def vote_metrics(logits: np.ndarray, votes: np.ndarray) -> dict:
    """Cross-entropy against vote shares; AUC for the annotators' majority class.

    Ties (a vote share of exactly 0.5) have no majority and are left out of AUC.
    """
    logits, votes = logits.astype(np.float64), votes.astype(np.float64)
    result = {"n": len(votes)}
    for index, name in enumerate(("valence", "arousal")):
        z, share = logits[:, index], votes[:, index]
        bce = np.logaddexp(0, z) - share * z  # -[s log p + (1 - s) log(1 - p)]
        decided = share != 0.5
        result[name] = {
            "bce": float(bce.mean()),
            "auc": auc(z[decided], (share[decided] > 0.5).astype(int)),
        }
    result["mean_bce"] = (result["valence"]["bce"] + result["arousal"]["bce"]) / 2
    return result


@torch.no_grad()
def predict(model: ContextPose, loader: DataLoader, device: torch.device) -> dict:
    """Model output plus any residual offset, with labels and participants.

    `pred` is on the label scale for regression and log-odds for `votes`;
    `votes` (vote shares) is None unless the dataset provides them.
    """
    model.eval()
    predictions, targets, votes, participants = [], [], [], []
    for batch in loader:
        pred = model(
            **{key: value.to(device) for key, value in batch["inputs"].items()}
        ) + batch["offset"].to(device)
        if not torch.isfinite(pred).all():
            raise ValueError("model produced non-finite predictions")
        predictions.append(pred.cpu().numpy())
        targets.append(batch["target"].numpy())
        if "votes" in batch:
            votes.append(batch["votes"].numpy())
        participants.extend(batch["participant_id"])
    return {
        "pred": np.concatenate(predictions),
        "target": np.concatenate(targets),
        "votes": np.concatenate(votes) if votes else None,
        "participants": participants,
    }


def evaluate(model: ContextPose, loader: DataLoader, device: torch.device) -> dict:
    return score_predictions(predict(model, loader, device), model.config.objective)


def score_predictions(out: dict, objective: str) -> dict:
    """Aggregate and per-participant metrics for `predict` output."""
    pred, ids = out["pred"], np.asarray(out["participants"])
    if objective == "votes":
        reference, metrics = out["votes"], vote_metrics
    else:
        reference, metrics = out["target"], regression_metrics
    return {
        "aggregate": metrics(pred, reference),
        "participants": {
            participant: metrics(
                pred[ids == participant], reference[ids == participant]
            )
            for participant in sorted(set(out["participants"]))
        },
    }


def selection_score(validation: dict) -> float:
    """Early-stopping criterion: mean MAE (regression) or mean cross-entropy."""
    aggregate = validation["aggregate"]
    return aggregate["mean_bce"] if "mean_bce" in aggregate else aggregate["mean_mae"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument(
        "--evaluate",
        type=Path,
        nargs="+",
        metavar="CHECKPOINT",
        help="score saved checkpoints on --test-manifest; several form an ensemble",
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        default="text_attention",
        help="context: CLIP text attention, or annotation FiLM/attention",
    )
    parser.add_argument("--text-dim", type=int)
    parser.add_argument("--num-joints", type=int)
    parser.add_argument("--joint-features", type=int)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--loss",
        choices=("mse", "l1"),
        default="mse",
        help="per-target squared or absolute error (L1 targets the median, like MAE)",
    )
    parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        default="regression",
        help="votes: log-odds of positive valence / high arousal votes (cross-entropy)",
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument("--inputs", choices=INPUTS, default="all")
    parser.add_argument("--text-skip", action="store_true")
    parser.add_argument(
        "--face-dim",
        type=int,
        default=0,
        help="per-step face feature width in the NPZs (prepare --face); 0 = no face",
    )
    parser.add_argument("--face-targets", choices=FACE_TARGETS, default="valence")
    parser.add_argument(
        "--face-dropout",
        type=float,
        default=0.0,
        help="chance of hiding a training sample's face stream",
    )
    parser.add_argument(
        "--augment", action="store_true", help="random rotate/scale/jitter/mirror"
    )
    parser.add_argument(
        "--residual-by",
        metavar="COLUMN",
        help="predict on top of the training median per value of this manifest column",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.patience) < 1:
        parser.error("epochs, batch-size and patience must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)

    if args.evaluate:
        if not args.test_manifest or args.train_manifest or args.val_manifest:
            parser.error(
                "--evaluate requires --test-manifest without train/val manifests"
            )
        if any(args.output.resolve() == path.resolve() for path in args.evaluate):
            parser.error("evaluation output must not overwrite a checkpoint")
        outputs, fits, objectives = [], [], set()
        for path in args.evaluate:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            config = ModelConfig(**checkpoint["config"])
            objectives.add(config.objective)
            test = PoseDataset(args.test_manifest, config, "test")
            if test.joint_order != checkpoint["joint_order"]:
                raise ValueError("test joint_order differs from checkpoint")
            for groups in checkpoint["selection_groups"].values():
                assert_disjoint(groups, test.groups())
            model = build_model(config).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            loader = DataLoader(
                WithOffset(test, checkpoint.get("residual")),
                batch_size=args.batch_size,
            )
            outputs.append(predict(model, loader, device))
            fits.append(checkpoint.get("classification"))
        if len(objectives) > 1:
            raise ValueError("ensemble checkpoints must share one objective")
        ensemble = {**outputs[0], "pred": np.mean([o["pred"] for o in outputs], 0)}
        report = score_predictions(ensemble, objectives.pop())
        if annotator_votes.has_votes(test.rows) and all(fits):
            report["classification"] = annotator_votes.ensemble_report(
                test.rows, [o["pred"] for o in outputs], fits
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report["aggregate"], allow_nan=False))
        return

    if not args.train_manifest or not args.val_manifest:
        parser.error("training requires --train-manifest and --val-manifest")
    if args.output == args.output.with_suffix(".metrics.json"):
        parser.error("checkpoint output must not use the .metrics.json suffix")
    if args.text_dim is None and args.model != "text_attention":
        args.text_dim = CONTEXT_DIM
    if any(
        value is None for value in (args.text_dim, args.num_joints, args.joint_features)
    ):
        parser.error("training requires --text-dim, --num-joints and --joint-features")
    if args.weight_decay < 0:
        parser.error("weight-decay must be non-negative")
    config = ModelConfig(
        args.text_dim,
        args.num_joints,
        args.joint_features,
        args.steps,
        width=args.width,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        inputs=args.inputs,
        text_skip=args.text_skip,
        model=args.model,
        face_dim=args.face_dim,
        face_targets=args.face_targets,
        face_dropout=args.face_dropout,
        objective=args.objective,
    )
    train = PoseDataset(args.train_manifest, config, "train")
    val = PoseDataset(args.val_manifest, config, "val")
    test = (
        PoseDataset(args.test_manifest, config, "test") if args.test_manifest else None
    )
    assert_group_separation(train, val, *([test] if test is not None else []))
    print(
        f"Eligible train={len(train)}, val={len(val)}; excluded needs_review="
        f"{train.excluded_review + val.excluded_review}"
    )
    residual = None
    if args.residual_by:
        key = "votes" if args.objective == "votes" else "target"
        targets = np.stack([train[i][key].numpy() for i in range(len(train))])
        group = group_logits if args.objective == "votes" else group_medians
        residual = group(train.rows, targets, args.residual_by)
    mirror = mirror_index(train.joint_order) if args.augment else None
    augmenter = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(
        WithOffset(train, residual),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(WithOffset(val, residual), batch_size=args.batch_size)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best, stale = float("inf"), 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            inputs = batch["inputs"]
            if args.augment:
                inputs = augment(inputs, mirror, augmenter)
            pred = model(**{key: value.to(device) for key, value in inputs.items()})
            pred = pred + batch["offset"].to(device)
            if args.objective == "votes":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    pred, batch["votes"].to(device), reduction="none"
                )
            else:
                error = pred - batch["target"].to(device)
                loss = error.abs() if args.loss == "l1" else error.square()
            loss = loss.mean(dim=0).sum()
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(pred)
        validation = evaluate(model, val_loader, device)
        score = selection_score(validation)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train),
                "validation": validation,
            }
        )
        print(
            f"epoch={epoch} train_loss={total_loss / len(train):.6f} val_score={score:.6f}"
        )
        if score < best:
            best, stale = score, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "seed": args.seed,
                    "validation": validation,
                    "joint_order": train.joint_order,
                    "residual": residual,
                    "selection_groups": {"train": train.groups(), "val": val.groups()},
                },
                args.output,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    if annotator_votes.has_votes(train.rows):
        # Binary-call thresholds, fitted on the clips used to fit and select.
        seen = [
            predict(
                model,
                DataLoader(WithOffset(part, residual), batch_size=args.batch_size),
                device,
            )
            for part in (train, val)
        ]
        checkpoint["classification"] = annotator_votes.fit_calls(
            train.rows + val.rows,
            np.concatenate([out["pred"] for out in seen]),
            args.residual_by or "event_text",
        )
        torch.save(checkpoint, args.output)
    report = {
        "best_epoch": checkpoint["epoch"],
        "validation": checkpoint["validation"],
        "history": history,
    }
    if test is not None:
        out = predict(
            model,
            DataLoader(WithOffset(test, residual), batch_size=args.batch_size),
            device,
        )
        report["test"] = score_predictions(out, config.objective)
        if "classification" in checkpoint and annotator_votes.has_votes(test.rows):
            report["test"]["classification"] = annotator_votes.ensemble_report(
                test.rows, [out["pred"]], [checkpoint["classification"]]
            )
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
