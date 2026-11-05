"""Linear probe: does simple movement carry valence/arousal signal?

Hand-crafted movement features for each window (pre, event, post), measured on
the 25 fps poses, feed linear models fitted with squared loss (ridge) or
absolute loss (least absolute deviations, which targets the median like MAE).
Context (the event-indicator combination), pose, and both are compared under
the same participant-grouped folds as `crossval`: the penalty is chosen on the
next fold, then refitted on both. Comparing context + pose with context alone
measures what pose adds without relying on a network's capacity to find it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .crossval import assign_folds, bootstrap_gain, read_rows
from .train import group_medians, regression_metrics

REGIONS = ("pre", "event", "post")
FEATURES = (
    "coverage",
    "head_speed",
    "elbow_speed",
    "wrist_speed",
    "wrist_speed_p90",
    "body_speed",
    "wrist_height",
    "wrist_spread",
    "head_bob",
)
HEAD, SHOULDERS, ELBOWS, WRISTS = [0], [5, 6], [7, 8], [9, 10]
MAX_SPEED = 20.0  # shoulder widths per second; faster steps are tracking jumps
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SETS = ("constant", "context", "pose", "context+pose")


def masked_mean(values: np.ndarray, ok: np.ndarray, axis=None) -> np.ndarray:
    count = ok.sum(axis)
    total = np.where(ok, values, 0.0).sum(axis)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def region_features(clip: dict, start: float, end: float) -> dict[str, float]:
    """Movement in [start, end) s; shoulder-width units, NaN when unobserved."""
    times = clip["frame_time_s"]
    keep = (times >= start) & (times < end)
    norm, valid, times = clip["keypoints_norm"][keep], clip["valid"][keep], times[keep]
    root = clip["root_px"][keep] / float(clip["scale_px"])
    body = valid[:, SHOULDERS].all(1)
    features = dict.fromkeys(FEATURES, np.nan)
    features["coverage"] = float(body.mean()) if len(body) else 0.0
    if len(times) < 2:
        return features
    dt = np.diff(times)[:, None]

    def speed(joints: list[int]) -> np.ndarray:
        ok = valid[1:, joints] & valid[:-1, joints]
        step = np.linalg.norm(np.diff(norm[:, joints], axis=0), axis=-1) / dt
        per_step = masked_mean(np.minimum(step, MAX_SPEED), ok, 1)
        return per_step[np.isfinite(per_step)]

    for name, joints in (("head", HEAD), ("elbow", ELBOWS), ("wrist", WRISTS)):
        steps = speed(joints)
        if len(steps):
            features[f"{name}_speed"] = float(steps.mean())
            if name == "wrist":
                features["wrist_speed_p90"] = float(np.percentile(steps, 90))
    moving = body[1:] & body[:-1]
    if moving.any():
        step = np.linalg.norm(np.diff(root, axis=0), axis=-1) / dt[:, 0]
        features["body_speed"] = float(np.minimum(step, MAX_SPEED)[moving].mean())
    wrists = valid[:, WRISTS]
    if wrists.any():
        positions = norm[:, WRISTS][wrists]
        features["wrist_height"] = float(positions[:, 1].mean())  # +y is down
        features["wrist_spread"] = float(positions.std(0).mean())
    if valid[:, 0].sum() > 1:
        features["head_bob"] = float(norm[valid[:, 0], 0, 1].std())
    return features


def clip_features(row: dict, poses: Path) -> np.ndarray:
    """[pre, event, post] x FEATURES from the clip's 25 fps pose file."""
    # data/<split>/clip.npz -> raw poses in train/ (train and val) or test/
    source = "test" if Path(row["path"]).parent.name == "test" else "train"
    with np.load(poses / source / Path(row["path"]).name) as data:
        clip = dict(data)
    start, end = float(row["event_start"]), float(row["event_end"])
    bounds = {
        "pre": (start - 5.0, start),
        "event": (start, end),
        "post": (float(row["post_start"]), float(row["post_end"])),
    }
    return np.array(
        [region_features(clip, *bounds[r])[f] for r in REGIONS for f in FEATURES]
    )


def fit_predict(x_train, y_train, x_test, alpha: float, loss: str) -> np.ndarray:
    """Penalised linear fit per target; `l1` is iteratively reweighted LAD."""
    mean, std = np.nanmean(x_train, 0), np.nanstd(x_train, 0)
    std = np.where(std > 0, std, 1.0)

    def design(x):
        z = np.nan_to_num((x - mean) / std)  # missing -> training mean
        return np.c_[np.ones(len(z)), z]

    a, b = design(x_train), design(x_test)
    penalty = alpha * np.eye(a.shape[1])
    penalty[0, 0] = 0.0  # never shrink the intercept
    predictions = []
    for target in y_train.T:
        weights = np.ones(len(a))
        for _ in range(100 if loss == "l1" else 1):
            coef = np.linalg.solve(
                a.T @ (weights[:, None] * a) + penalty, a.T @ (weights * target)
            )
            weights = 1.0 / np.maximum(np.abs(target - a @ coef), 1e-3)
        predictions.append(b @ coef)
    return np.stack(predictions, 1)


def load_clips(rows: list[dict]) -> tuple[np.ndarray, list[tuple]]:
    targets, contexts = [], []
    for row in rows:
        with np.load(row["path"]) as sample:
            targets.append(sample["target"])
            contexts.append(tuple(sample["context"].tolist()))
    return np.stack(targets), contexts


def evaluate_once(manifest, poses, dev_x, dev_y, dev_contexts, folds, output):
    """Fit the pre-specified candidate on all development clips; score test once.

    Candidate: valence = median per indicator combination; arousal = ridge on
    context one-hot + movement features, its penalty chosen by out-of-fold
    arousal MAE on the development folds. Baselines are scored alongside.
    """
    rows = [r for r in read_rows([manifest])[0] if r["needs_review"] != "true"]
    if any(row["split"] != "test" for row in rows):
        raise SystemExit("--test-manifest must contain only test rows")
    y, contexts = load_clips(rows)
    combos = sorted(set(dev_contexts))
    context = np.array([[c == combo for combo in combos] for c in contexts], float)
    x = np.c_[context, np.stack([clip_features(r, poses) for r in rows])]

    def arousal_error(alpha: float) -> float:
        errors = []
        for fold in np.unique(folds):
            seen, held = folds != fold, folds == fold
            pred = fit_predict(dev_x[seen], dev_y[seen], dev_x[held], alpha, "l2")
            errors.append(np.abs(pred[:, 1] - dev_y[held, 1]))
        return float(np.concatenate(errors).mean())

    alpha = min(ALPHAS, key=arousal_error)
    ridge = fit_predict(dev_x, dev_y, x, alpha, "l2")
    table = group_medians([{"combo": str(c)} for c in dev_contexts], dev_y, "combo")
    lookup = np.array([table["groups"].get(str(c), table["default"]) for c in contexts])
    predictions = {
        "candidate": np.c_[lookup[:, 0], ridge[:, 1]],
        "combination_median": lookup,
        "overall_median": np.tile(np.median(dev_y, 0), (len(y), 1)),
        "ridge_context+pose": ridge,
    }
    groups = np.array([r["participant_id"] for r in rows])
    report = {
        "n": len(rows),
        "participants": len(set(groups)),
        "arousal_alpha": alpha,
        "results": {k: regression_metrics(p, y) for k, p in predictions.items()},
        "candidate_gains": {},
    }
    candidate = np.abs(predictions["candidate"] - y)
    for base in ("combination_median", "overall_median"):
        base_err = np.abs(predictions[base] - y)
        for target, columns in (("mean", slice(None)), ("arousal", slice(1, 2))):
            model_err, other = (
                candidate[:, columns].mean(1),
                base_err[:, columns].mean(1),
            )
            report["candidate_gains"][f"vs {base}, {target}"] = {
                "mae_gain": float((other - model_err).mean()),
                "ci95": bootstrap_gain(groups, model_err, other),
            }
    (output / "test_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nTEST (once): {report['n']} clips, {report['participants']} videos")
    for name, m in report["results"].items():
        v, a = m["valence"], m["arousal"]
        print(
            f"{name:20} mean MAE {m['mean_mae']:.3f} | valence {v['mae']:.3f}"
            f" / {v['rmse']:.3f} / {v['ccc'] or 0:+.2f} | arousal {a['mae']:.3f}"
            f" / {a['rmse']:.3f} / {a['ccc'] or 0:+.2f}"
        )
    for key, gain in report["candidate_gains"].items():
        low, high = gain["ci95"]
        print(
            f"candidate {key}: MAE gain {gain['mae_gain']:+.3f} [{low:+.3f}, {high:+.3f}]"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--poses", type=Path, default=Path("dataset/ViTPoseFilled"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/linear_probe"))
    parser.add_argument(
        "--test-manifest",
        type=Path,
        help="score the pre-specified candidate once on held-out test clips",
    )
    args = parser.parse_args(argv)

    rows = [r for r in read_rows(args.manifests)[0] if r["needs_review"] != "true"]
    if any(row["split"] == "test" for row in rows):
        parser.error("the linear probe must not use test manifests")
    fold_of = assign_folds(rows, args.folds, args.fold_seed)
    folds = np.array([fold_of[r["participant_id"]] for r in rows])
    pose = np.stack([clip_features(r, args.poses) for r in rows])
    y, contexts = load_clips(rows)
    combos = sorted(set(contexts))
    context = np.array([[c == combo for combo in combos] for c in contexts], float)
    inputs = {
        "constant": np.zeros((len(rows), 0)),
        "context": context,
        "pose": pose,
        "context+pose": np.c_[context, pose],
    }

    predictions = {}
    for loss in ("l2", "l1"):
        for name, x in inputs.items():
            pred = np.zeros_like(y)
            for fold in range(args.folds):
                inner = (fold + 1) % args.folds
                train = (folds != fold) & (folds != inner)
                errors = [
                    np.abs(
                        fit_predict(x[train], y[train], x[folds == inner], a, loss)
                        - y[folds == inner]
                    ).mean()
                    for a in ALPHAS
                ]
                alpha = ALPHAS[int(np.argmin(errors))]
                seen = folds != fold
                pred[folds == fold] = fit_predict(
                    x[seen], y[seen], x[folds == fold], alpha, loss
                )
            predictions[(loss, name)] = pred

    groups = np.array([r["participant_id"] for r in rows])
    report = {"n": len(rows), "results": {}, "gains": {}, "correlations": {}}
    for (loss, name), pred in predictions.items():
        report["results"][f"{loss}/{name}"] = regression_metrics(pred, y)
    targets = {"mean": slice(None), "valence": slice(0, 1), "arousal": slice(1, 2)}
    for loss in ("l2", "l1"):
        for model, base in (("context+pose", "context"), ("pose", "constant")):
            for target, columns in targets.items():
                model_err, base_err = (
                    np.abs(predictions[(loss, k)] - y)[:, columns].mean(1)
                    for k in (model, base)
                )
                report["gains"][f"{loss}: {model} vs {base}, {target}"] = {
                    "mae_gain": float((base_err - model_err).mean()),
                    "ci95": bootstrap_gain(groups, model_err, base_err),
                }
    names = [f"{r}_{f}" for r in REGIONS for f in FEATURES]
    for index, target in enumerate(("valence", "arousal")):
        rho = {}
        for column, name in enumerate(names):
            ok = np.isfinite(pose[:, column])
            rho[name] = float(spearmanr(pose[ok, column], y[ok, index])[0])
        report["correlations"][target] = dict(
            sorted(rho.items(), key=lambda item: -abs(item[1]))
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"{len(rows)} clips, {args.folds} participant-grouped folds")
    for key, m in report["results"].items():
        v, a = m["valence"], m["arousal"]
        print(
            f"{key:16} mean MAE {m['mean_mae']:.3f} | valence {v['mae']:.3f}"
            f" / {v['rmse']:.3f} / {v['ccc'] or 0:+.2f} | arousal {a['mae']:.3f}"
            f" / {a['rmse']:.3f} / {a['ccc'] or 0:+.2f}"
        )
    for key, gain in report["gains"].items():
        low, high = gain["ci95"]
        print(f"{key}: MAE gain {gain['mae_gain']:+.3f} [{low:+.3f}, {high:+.3f}]")
    for target, rho in report["correlations"].items():
        top = ", ".join(f"{k} {v:+.2f}" for k, v in list(rho.items())[:4])
        print(f"strongest {target} correlations: {top}")
    if args.test_manifest:
        evaluate_once(
            args.test_manifest,
            args.poses,
            inputs["context+pose"],
            y,
            contexts,
            folds,
            args.output,
        )


if __name__ == "__main__":
    main()
