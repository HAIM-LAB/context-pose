import json

import numpy as np
import pytest

from asd_contextpose.regression_table import (
    average_precision,
    config_key,
    constant_share,
    main,
    score,
    streams_of,
)

BASE = ["--objective", "votes", "--num-joints", "11", "--joint-features", "4"]


@pytest.mark.parametrize(
    "extra, expected",
    [
        (["--inputs", "pose"], "P"),
        (["--inputs", "text", "--model", "annotation_attention",
          "--residual-by", "event_text"], "A"),  # fmt: skip
        (["--inputs", "text", "--text-dim", "384"], "V"),
        (["--inputs", "face", "--face-dim", "768"], "F"),
        (["--inputs", "face", "--face-dim", "768", "--residual-by", "event_text"], "AF"),
        (["--model", "annotation_attention", "--face-dim", "768",
          "--residual-by", "event_text"], "PAF"),  # fmt: skip
        (["--text-dim", "384", "--residual-by", "event_text"], "PAV"),
        (["--inputs", "text", "--text-dim", "384", "--face-dim", "768",
          "--residual-by", "event_text"], "AVF"),  # fmt: skip
    ],
)
def test_streams_follow_the_training_arguments(extra, expected):
    assert streams_of(BASE + extra) == expected


def test_seed_is_ignored_when_grouping():
    assert config_key(BASE + ["--seed", "1"]) == config_key(BASE + ["--seed=2"])
    assert config_key(BASE + ["--seed", "1"]) != config_key(BASE + ["--lr", "1e-3"])


def test_average_precision_matches_hand_computation():
    # ranked labels 1, 0, 1, 0: precision 1/1 at the first hit, 2/3 at the second
    assert average_precision(np.array([0.9, 0.8, 0.7, 0.1]), np.array([1, 0, 1, 0])) == (
        pytest.approx((1 + 2 / 3) / 2)
    )
    assert average_precision(np.array([0.5, 0.5]), np.array([0, 0])) is None


def test_perfect_prediction_scores_zero_error():
    share = np.array([[0.0, 1.0], [1 / 3, 2 / 3], [1.0, 0.0], [2 / 3, 1 / 3]])
    classes = (share > 0.5).astype(int)
    metrics = score(share, share, classes)
    assert metrics["mean_mae"] == metrics["mean_rmse"] == 0
    assert metrics["valence"]["ccc"] == pytest.approx(1)
    assert metrics["map"] == 1


def test_constant_uses_only_the_other_folds():
    share = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    fold = np.array([0, 1, 2])
    np.testing.assert_allclose(constant_share(share, fold)[0], [0.75, 0.75])


def write_run(folder, args, seed, rng, shares):
    preds = []
    for i, share in enumerate(shares):
        logit = np.log(np.clip(share, 0.05, 0.95) / (1 - np.clip(share, 0.05, 0.95)))
        preds.append(
            {
                "fold": i % 5,
                "path": f"clip{i}.npz",
                "participant_id": f"p{i}",
                "pred": (logit + rng.normal(0, 0.5, 2)).tolist(),
                "votes": share.tolist(),
                "classes": [int(s > 0.5) if s != 0.5 else -1 for s in share],
                "scores": {"lookup": [0.6, 0.3]},
            }
        )
    folder.mkdir(parents=True)
    report = {"objective": "votes", "train_args": args + ["--seed", str(seed)],
              "predictions": preds}  # fmt: skip
    (folder / "cv_report.json").write_text(json.dumps(report))


def test_end_to_end_table(tmp_path, capsys):
    rng = np.random.default_rng(0)
    shares = rng.choice([0, 1 / 3, 2 / 3, 1], size=(40, 2))
    for name, extra in {
        "T_P_pose": ["--inputs", "pose"],
        "T_PAF_full": ["--model", "annotation_attention", "--face-dim", "768",
                       "--residual-by", "event_text"],  # fmt: skip
    }.items():
        for seed in range(1, 6):
            write_run(tmp_path / "streams" / name / f"seed{seed}", BASE + extra, seed,
                      rng, shares)  # fmt: skip
    out = tmp_path / "table.tex"
    main([str(tmp_path / "streams"), "--output", str(out), "--json", str(tmp_path / "m.json")])
    table = out.read_text()
    assert "Pose only" in table and "Pose + annotation + face" in table
    assert "Event-annotation lookup" in table and "Training-fold mean share" in table
    assert table.index("Pose only") < table.index("Pose + annotation + face")
    summary = json.loads((tmp_path / "m.json").read_text())
    assert set(summary) == {"P", "PAF", "references"}
    assert "seeds=5" in capsys.readouterr().out
