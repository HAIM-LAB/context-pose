import csv

import numpy as np
import pytest

from asd_contextpose.data import PoseDataset, assert_group_separation


def test_load_and_separate_inputs(make_manifest, config):
    dataset = PoseDataset(make_manifest(), config, "train")
    sample = dataset[0]
    assert set(sample["inputs"]) == {
        "text_embedding",
        "pre_pose",
        "event_pose",
        "post_pose",
    }
    assert sample["target"].shape == (2,)
    assert len(dataset) == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"target": np.zeros(3, dtype=np.float32)},
        {"target": np.zeros(2, dtype=np.float64)},
        {"target": np.array([0, np.nan], dtype=np.float32)},
        {"text_embedding": np.array([object()], dtype=object)},
        {"pre_pose": np.zeros((31, 3, 3), dtype=np.float32)},
    ],
)
def test_invalid_arrays(make_manifest, config, overrides):
    with pytest.raises(ValueError):
        PoseDataset(make_manifest(overrides=overrides), config, "train")


@pytest.mark.parametrize("field", ["participant", "recording"])
def test_reject_cross_split_group(make_manifest, config, field):
    train = PoseDataset(make_manifest(**{field: "same"}), config, "train")
    val = PoseDataset(make_manifest("val", **{field: "same"}), config, "val")
    with pytest.raises(ValueError, match="split leakage"):
        assert_group_separation(train, val)


def test_accept_disjoint_groups(make_manifest, config):
    datasets = [
        PoseDataset(make_manifest(split), config, split)
        for split in ("train", "val", "test")
    ]
    assert_group_separation(*datasets)


def test_exclude_review_before_loading(make_manifest, config):
    manifest = make_manifest(metadata={"needs_review": "false"})
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0].update(needs_review="true", path="missing.npz")
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dataset = PoseDataset(manifest, config, "train")
    assert len(dataset) == 1
    assert dataset.excluded_review == 1


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"needs_review": "true"}, "no eligible"),
        ({"needs_review": "maybe"}, "needs_review"),
        ({"joint_order": "a|b"}, "joint_order"),
        ({"post_start": "3"}, "together"),
        ({"event_start": "2", "event_end": "1"}, "invalid event"),
        ({"event_start": "2", "event_end": "4", "event_duration": "3"}, "disagrees"),
        (
            {"event_start": "2", "event_end": "4", "post_start": "3", "post_end": "8"},
            "at or after",
        ),
        ({"outcome_start": "4", "outcome_end": "6"}, "requires post"),
        (
            {
                "post_start": "3",
                "post_end": "8",
                "outcome_start": "4",
                "outcome_end": "6",
            },
            "overlaps outcome",
        ),
    ],
)
def test_metadata_validation(make_manifest, config, metadata, message):
    with pytest.raises(ValueError, match=message):
        PoseDataset(make_manifest(metadata=metadata), config, "train")


def test_joint_order_across_splits(make_manifest, config):
    train = PoseDataset(
        make_manifest(metadata={"joint_order": "a|b|c"}), config, "train"
    )
    val = PoseDataset(
        make_manifest("val", metadata={"joint_order": "b|a|c"}), config, "val"
    )
    with pytest.raises(ValueError, match="joint_order"):
        assert_group_separation(train, val)


def test_split_mismatch(make_manifest, config):
    with pytest.raises(ValueError, match="expected split"):
        PoseDataset(make_manifest(), config, "val")


def test_truncated_csv_row(make_manifest, config):
    manifest = make_manifest()
    manifest.write_text("path,participant_id,recording_id,split\na.npz,p\n")
    with pytest.raises(ValueError, match="number of fields"):
        PoseDataset(manifest, config, "train")


def test_missing_array(make_manifest, config):
    manifest = make_manifest()
    np.savez(manifest.parent / "0.npz", target=np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="missing "):
        PoseDataset(manifest, config, "train")
