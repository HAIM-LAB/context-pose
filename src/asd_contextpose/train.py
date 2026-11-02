"""Small CPU-first training/evaluation CLI; no text or video processing."""

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PoseDataset, assert_disjoint, assert_group_separation
from .model import ContextPose, ModelConfig


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


@torch.no_grad()
def evaluate(model: ContextPose, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    predictions, targets, participants = [], [], []
    for batch in loader:
        pred = model(
            **{key: value.to(device) for key, value in batch["inputs"].items()}
        )
        if not torch.isfinite(pred).all():
            raise ValueError("model produced non-finite predictions")
        predictions.append(pred.cpu().numpy())
        targets.append(batch["target"].numpy())
        participants.extend(batch["participant_id"])
    pred, target = np.concatenate(predictions), np.concatenate(targets)
    ids = np.asarray(participants)
    return {
        "aggregate": regression_metrics(pred, target),
        "participants": {
            participant: regression_metrics(
                pred[ids == participant], target[ids == participant]
            )
            for participant in sorted(set(participants))
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument("--evaluate", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--text-dim", type=int)
    parser.add_argument("--num-joints", type=int)
    parser.add_argument("--joint-features", type=int)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
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
        if args.output.resolve() == args.evaluate.resolve():
            parser.error("evaluation output must not overwrite the checkpoint")
        checkpoint = torch.load(args.evaluate, map_location="cpu", weights_only=True)
        config = ModelConfig(**checkpoint["config"])
        test = PoseDataset(args.test_manifest, config, "test")
        if test.joint_order != checkpoint["joint_order"]:
            raise ValueError("test joint_order differs from checkpoint")
        for groups in checkpoint["selection_groups"].values():
            assert_disjoint(groups, test.groups())
        model = ContextPose(config).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        report = evaluate(model, DataLoader(test, batch_size=args.batch_size), device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report["aggregate"], allow_nan=False))
        return

    if not args.train_manifest or not args.val_manifest:
        parser.error("training requires --train-manifest and --val-manifest")
    if args.output == args.output.with_suffix(".metrics.json"):
        parser.error("checkpoint output must not use the .metrics.json suffix")
    if any(
        value is None for value in (args.text_dim, args.num_joints, args.joint_features)
    ):
        parser.error("training requires --text-dim, --num-joints and --joint-features")
    config = ModelConfig(
        args.text_dim, args.num_joints, args.joint_features, args.steps
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
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val, batch_size=args.batch_size)
    model = ContextPose(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best, stale = float("inf"), 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(
                **{key: value.to(device) for key, value in batch["inputs"].items()}
            )
            loss = (pred - batch["target"].to(device)).square().mean(dim=0).sum()
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(pred)
        validation = evaluate(model, val_loader, device)
        score = validation["aggregate"]["mean_mae"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train),
                "validation": validation,
            }
        )
        print(
            f"epoch={epoch} train_loss={total_loss / len(train):.6f} val_mean_mae={score:.6f}"
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
                    "selection_groups": {"train": train.groups(), "val": val.groups()},
                },
                args.output,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    report = {
        "best_epoch": checkpoint["epoch"],
        "validation": checkpoint["validation"],
        "history": history,
    }
    if test is not None:
        model.load_state_dict(checkpoint["state_dict"])
        report["test"] = evaluate(
            model, DataLoader(test, batch_size=args.batch_size), device
        )
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
