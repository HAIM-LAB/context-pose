import json

import numpy as np
import pytest
import torch

from asd_contextpose.train import (
    POSE_KEYS,
    augment,
    group_logits,
    group_medians,
    main,
    mirror_index,
    regression_metrics,
)


def test_metrics():
    target = np.array([[0.0, 1.0], [2.0, 3.0]])
    exact = regression_metrics(target, target)
    assert exact["mean_mae"] == 0
    assert exact["valence"]["ccc"] == pytest.approx(1)
    offset = regression_metrics(target + 1, target)
    assert offset["arousal"]["mae"] == pytest.approx(1)
    assert offset["arousal"]["rmse"] == pytest.approx(1)
    assert offset["arousal"]["ccc"] == pytest.approx(2 / 3)
    assert regression_metrics(target[:1], target[:1])["valence"]["ccc"] is None
    assert (
        regression_metrics(np.ones((2, 2)), np.ones((2, 2)))["valence"]["ccc"] is None
    )


def test_training_and_evaluation_cli(make_manifest, tmp_path):
    train, val, test = [make_manifest(split) for split in ("train", "val", "test")]
    output = tmp_path / "best.pt"
    main(
        [
            "--train-manifest",
            str(train),
            "--val-manifest",
            str(val),
            "--test-manifest",
            str(test),
            "--text-dim",
            "8",
            "--num-joints",
            "3",
            "--joint-features",
            "3",
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--output",
            str(output),
        ]
    )
    checkpoint = torch.load(output, weights_only=True)
    report = json.loads(output.with_suffix(".metrics.json").read_text())
    assert checkpoint["config"]["text_dim"] == 8
    assert report["best_epoch"] in (1, 2)
    best = min(
        report["history"], key=lambda row: row["validation"]["aggregate"]["mean_mae"]
    )
    assert report["best_epoch"] == best["epoch"]
    assert "p-val" in report["validation"]["participants"]
    assert "p-test" in report["test"]["participants"]
    evaluation = tmp_path / "evaluation.json"
    main(
        [
            "--evaluate",
            str(output),
            "--test-manifest",
            str(test),
            "--output",
            str(evaluation),
        ]
    )
    assert json.loads(evaluation.read_text()) == report["test"]
    leaked_test = make_manifest("test", participant="p-train")
    with pytest.raises(ValueError, match="participant_id"):
        main(
            [
                "--evaluate",
                str(output),
                "--test-manifest",
                str(leaked_test),
                "--output",
                str(evaluation),
            ]
        )


def test_early_stopping_preserves_best(make_manifest, tmp_path, monkeypatch):
    scores = iter([0.5, 0.6, 0.7])
    monkeypatch.setattr(
        "asd_contextpose.train.evaluate",
        lambda *args: {"aggregate": {"mean_mae": next(scores)}, "participants": {}},
    )
    output = tmp_path / "early.pt"
    main(
        [
            "--train-manifest",
            str(make_manifest()),
            "--val-manifest",
            str(make_manifest("val")),
            "--text-dim",
            "8",
            "--num-joints",
            "3",
            "--joint-features",
            "3",
            "--epochs",
            "10",
            "--patience",
            "2",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.with_suffix(".metrics.json").read_text())
    assert len(report["history"]) == 3
    assert report["best_epoch"] == 1
    assert torch.load(output, weights_only=True)["epoch"] == 1


def test_mirror_index_swaps_left_and_right():
    order = "nose|left_eye|right_eye|left_wrist|right_wrist"
    assert mirror_index(order) == [0, 2, 1, 4, 3]
    assert mirror_index("a|b") == [0, 1]
    assert mirror_index("left_eye|nose") is None


def augment_inputs():
    poses = torch.randn(64, 6, 3, 3)
    poses[..., 2] = 1.0
    poses[:, :, 1] = 0.0  # missing joint: zero coordinates and confidence
    inputs = {"text_embedding": torch.zeros(64, 8)}
    inputs.update({k: poses[:, 2 * i : 2 * i + 2] for i, k in enumerate(POSE_KEYS)})
    return inputs, poses


def test_augment_moves_observed_joints_only():
    inputs, poses = augment_inputs()
    out = augment(inputs, None, torch.Generator().manual_seed(0))
    joined = torch.cat([out[k] for k in POSE_KEYS], dim=1)
    assert joined.shape == poses.shape
    assert (joined[..., 2] == poses[..., 2]).all()  # confidence kept
    assert (joined[:, :, 1] == 0).all()  # missing joint stays missing
    assert not torch.allclose(joined[:, :, 0, :2], poses[:, :, 0, :2])


def test_augment_mirror_swaps_joints_and_negates_x():
    inputs, poses = augment_inputs()
    out = augment(inputs, [0, 2, 1], torch.Generator().manual_seed(0))
    joined = torch.cat([out[k] for k in POSE_KEYS], dim=1)
    flipped = joined[:, 0, 2, 2] == 0  # the missing joint moved to index 2
    assert 0 < flipped.sum() < len(flipped)
    assert (joined[flipped][:, :, 2] == 0).all()
    assert (joined[~flipped][:, :, 1] == 0).all()


def test_group_medians_fall_back_for_small_groups():
    rows = [{"text": "a"}] * 3 + [{"text": "b"}]
    targets = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [-1.0, -1.0]])
    medians = group_medians(rows, targets, "text")
    assert medians["groups"] == {"a": [1.0, 0.0]}
    assert medians["default"] == [0.5, 0.0]


def test_residual_training_adds_the_group_median(make_manifest, tmp_path):
    train, val = (
        make_manifest(split, count=3, metadata={"group": "g"})
        for split in ("train", "val")
    )
    output = tmp_path / "best.pt"
    main(
        ["--train-manifest", str(train), "--val-manifest", str(val)]
        + ["--text-dim", "8", "--num-joints", "3", "--joint-features", "3"]
        + ["--epochs", "1", "--residual-by", "group", "--augment"]
        + ["--output", str(output)]
    )
    residual = torch.load(output, weights_only=True)["residual"]
    assert residual["column"] == "group" and set(residual["groups"]) == {"g"}


@pytest.mark.parametrize("variant", ["annotation_film", "annotation_attention"])
def test_annotation_models_train_without_clip(make_manifest, tmp_path, variant):
    context = {"context": np.array([1, 0, 1, 0], np.float32)}
    train, val, test = (
        make_manifest(split, overrides=context) for split in ("train", "val", "test")
    )
    output = tmp_path / "best.pt"
    main(
        ["--model", variant, "--train-manifest", str(train), "--val-manifest", str(val)]
        + ["--test-manifest", str(test), "--num-joints", "3", "--joint-features", "3"]
        + ["--epochs", "1", "--output", str(output)]
    )
    checkpoint = torch.load(output, weights_only=True)
    assert checkpoint["config"]["model"] == variant
    assert checkpoint["config"]["text_dim"] == 4
    report = json.loads(output.with_suffix(".metrics.json").read_text())
    assert report["test"]["aggregate"]["n"] == 2


def test_l1_loss_option(make_manifest, tmp_path):
    train, val = (make_manifest(split) for split in ("train", "val"))
    output = tmp_path / "best.pt"
    main(
        ["--train-manifest", str(train), "--val-manifest", str(val), "--loss", "l1"]
        + ["--text-dim", "8", "--num-joints", "3", "--joint-features", "3"]
        + ["--epochs", "2", "--output", str(output)]
    )
    report = json.loads(output.with_suffix(".metrics.json").read_text())
    assert len(report["history"]) >= 1


def test_face_stream_trains_from_npz_face_keys(make_manifest, tmp_path):
    rng = np.random.default_rng(0)
    face = {
        key: np.c_[rng.normal(size=(32, 4)), np.ones(32)].astype(np.float32)
        for key in ("pre_face", "event_face", "post_face")
    }
    train, val = (make_manifest(split, overrides=face) for split in ("train", "val"))
    output = tmp_path / "best.pt"
    main(
        ["--train-manifest", str(train), "--val-manifest", str(val)]
        + ["--text-dim", "8", "--num-joints", "3", "--joint-features", "3"]
        + ["--face-dim", "4", "--face-dropout", "0.3", "--face-targets", "both"]
        + ["--epochs", "1", "--output", str(output)]
    )
    config = torch.load(output, weights_only=True)["config"]
    assert config["face_dim"] == 4 and config["face_targets"] == "both"


def test_group_logits_are_log_odds_of_mean_vote_shares():
    rows = [{"text": "a"}] * 3 + [{"text": "b"}]
    votes = np.array([[0.5, 1.0], [0.5, 1.0], [0.5, 1.0], [1.0, 0.0]])
    table = group_logits(rows, votes, "text")
    assert set(table["groups"]) == {"a"}
    assert table["groups"]["a"][0] == pytest.approx(0.0)
    assert table["groups"]["a"][1] == pytest.approx(np.log(0.98 / 0.02))  # clipped


def test_votes_objective_trains_on_vote_shares(make_manifest, tmp_path):
    votes = {"votes": np.array([0.25, 0.75], np.float32)}
    train, val = (
        make_manifest(split, count=3, overrides=votes, metadata={"group": "g"})
        for split in ("train", "val")
    )
    output = tmp_path / "best.pt"
    main(
        ["--train-manifest", str(train), "--val-manifest", str(val)]
        + ["--text-dim", "8", "--num-joints", "3", "--joint-features", "3"]
        + ["--objective", "votes", "--residual-by", "group", "--epochs", "2"]
        + ["--output", str(output)]
    )
    checkpoint = torch.load(output, weights_only=True)
    assert checkpoint["config"]["objective"] == "votes"
    assert checkpoint["residual"]["groups"]["g"][0] == pytest.approx(np.log(1 / 3))
    report = json.loads(output.with_suffix(".metrics.json").read_text())
    assert "mean_bce" in report["validation"]["aggregate"]


def test_checkpoints_store_call_rules_and_ensembles_evaluate(make_manifest, tmp_path):
    meta = {"valence_votes": "1|1|0", "arousal_votes": "-1|-1|1"}
    votes = {"votes": np.array([2 / 3, 1 / 3], np.float32)}
    train, val, test = (
        make_manifest(split, count=3, metadata=meta, overrides=votes)
        for split in ("train", "val", "test")
    )
    paths = []
    for seed in ("1", "2"):
        path = tmp_path / f"seed{seed}.pt"
        main(
            ["--train-manifest", str(train), "--val-manifest", str(val)]
            + ["--test-manifest", str(test), "--objective", "votes", "--seed", seed]
            + ["--text-dim", "8", "--num-joints", "3", "--joint-features", "3"]
            + ["--epochs", "1", "--output", str(path)]
        )
        paths.append(path)
    rules = torch.load(paths[0], weights_only=True)["classification"]
    assert set(rules) == {"lookup", "model_thresholds", "lookup_thresholds", "majority"}
    single = tmp_path / "single.json"
    main(
        ["--evaluate", str(paths[0]), "--test-manifest", str(test)]
        + ["--output", str(single)]
    )
    trained = json.loads(paths[0].with_suffix(".metrics.json").read_text())["test"]
    assert json.loads(single.read_text()) == trained
    ensemble = tmp_path / "ensemble.json"
    main(
        ["--evaluate", *map(str, paths), "--test-manifest", str(test)]
        + ["--output", str(ensemble)]
    )
    report = json.loads(ensemble.read_text())
    assert "mean_bce" in report["aggregate"]
    assert report["classification"]["valence"]["n"] == 3
    assert report["classification"]["arousal"]["annotators"]["n"] == 3
