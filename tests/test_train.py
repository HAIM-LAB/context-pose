import json

import numpy as np
import pytest
import torch

from asd_contextpose.train import main, regression_metrics


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
