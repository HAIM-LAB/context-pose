import csv
import json

import numpy as np
import pytest

from asd_contextpose.crossval import assign_folds, main
from asd_contextpose.votes import best_threshold, vote_labels


def test_folds_keep_participants_together_and_balance_categories():
    rows = [
        {"participant_id": f"{category}-{n}", "category": category}
        for category in ("a", "b")
        for n in range(6)
        for _ in range(2)  # two clips per participant
    ]
    fold_of = assign_folds(rows, folds=3, seed=0)
    assert set(fold_of) == {row["participant_id"] for row in rows}
    for category in ("a", "b"):
        folds = [fold_of[f"{category}-{n}"] for n in range(6)]
        assert sorted(folds) == [0, 0, 1, 1, 2, 2]


def test_every_clip_is_scored_once_out_of_fold(make_manifest, tmp_path):
    manifests = [
        make_manifest(split, participant=f"p{n}", recording=f"r{n}")
        for n, split in enumerate(["train"] * 4 + ["val"] * 2)
    ]
    output = tmp_path / "cv"
    main(
        ["--manifests", *map(str, manifests), "--folds", "3", "--output", str(output)]
        + ["--text-dim", "8", "--num-joints", "3", "--joint-features", "3"]
        + ["--epochs", "1", "--batch-size", "2"]
    )
    report = json.loads((output / "cv_report.json").read_text())
    scored = [(p["participant_id"], p["path"]) for p in report["predictions"]]
    assert len(scored) == len(set(scored)) == 12  # 6 participants x 2 clips
    for p in report["predictions"]:
        for split in ("train", "val"):
            with (output / f"fold_{p['fold']}" / f"{split}.csv").open() as handle:
                seen = {row["participant_id"] for row in csv.DictReader(handle)}
            assert p["participant_id"] not in seen
    assert report["model"]["n"] == 12


def test_rejects_test_manifests(make_manifest, tmp_path):
    with pytest.raises(SystemExit):
        main(["--manifests", str(make_manifest("test")), "--output", str(tmp_path)])


def vote_manifests(make_manifest):
    """Six participants: even ones majority positive/high, odd ones not."""
    manifests = []
    for n, split in enumerate(["train"] * 4 + ["val"] * 2):
        high = n % 2 == 0
        votes = {
            "valence_votes": "1|1|0" if high else "0|0|1",
            "arousal_votes": "1|1|-1" if high else "-1|-1|1",
        }
        share = np.float32(2 / 3 if high else 1 / 3)
        manifests.append(
            make_manifest(
                split,
                participant=f"p{n}",
                recording=f"r{n}",
                metadata=votes,
                overrides={"votes": np.array([share, share])},
            )
        )
    return manifests


@pytest.mark.parametrize("objective", ["regression", "votes"])
def test_classification_report_scores_annotator_majorities(
    make_manifest, tmp_path, objective
):
    output = tmp_path / "cv"
    main(
        ["--manifests", *map(str, vote_manifests(make_manifest)), "--folds", "3"]
        + ["--output", str(output), "--objective", objective]
        + ["--text-dim", "8", "--num-joints", "3", "--joint-features", "3"]
        + ["--epochs", "1", "--batch-size", "2"]
    )
    report = json.loads((output / "cv_report.json").read_text())
    assert report["objective"] == objective
    assert ("mean_bce" in report["model"]) == (objective == "votes")
    for target in ("valence", "arousal"):
        entry = report["classification"][target]
        assert entry["n"] == 12 and entry["positives"] == 6
        assert entry["majority"]["balanced_accuracy"] == 0.5
        assert set(entry) >= {"model", "lookup", "annotators"}
        assert 0 <= entry["model"]["auc"] <= 1


def test_best_threshold_and_vote_labels():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert best_threshold(scores, np.array([0, 0, 1, 1])) == pytest.approx(0.5)
    shares, classes = vote_labels(
        [{"valence_votes": "1|0|", "arousal_votes": "1|1|-1"}]
    )
    np.testing.assert_allclose(shares, [[0.5, 2 / 3]])
    assert classes.tolist() == [[-1, 1]]  # valence tie, arousal high
