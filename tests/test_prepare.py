import csv

import numpy as np
import openpyxl
import pytest

from asd_contextpose import ModelConfig, prepare
from asd_contextpose.data import PoseDataset, assert_group_separation
from asd_contextpose.prepare import (
    INDICATORS,
    SINGLE_ANNOTATOR_COLUMNS,
    TEXT_CSV,
    convert_clip,
    load_single_annotator,
    main,
    resample,
)

JOINTS = ["nose", "left_wrist", "right_wrist"]


def write_clip(path, seconds, fps=25.0, valid=None):
    """Synthetic clip whose x coordinate equals the frame time, for boundary checks."""
    t = np.arange(round(seconds * fps), dtype=np.float32) / fps
    norm = np.stack([np.repeat(t[:, None], 3, 1), np.ones((len(t), 3))], -1)
    valid = np.ones((len(t), 3), bool) if valid is None else valid
    np.savez(
        path,
        keypoints_norm=np.where(valid[..., None], norm, np.nan).astype(np.float32),
        kp_conf=np.full((len(t), 3), 1.02, np.float32),
        valid=valid,
        fps=np.float32(fps),
        joint_names=np.array(JOINTS),
    )


def test_regions_follow_five_second_flanks(tmp_path):
    write_clip(tmp_path / "c.npz", 20)
    out = convert_clip(tmp_path / "c.npz", steps=32)
    assert out["pre_pose"][[0, -1], 0, 0] == pytest.approx([0, 4.96])
    assert out["event_pose"][[0, -1], 0, 0] == pytest.approx([5, 14.96])
    assert out["post_pose"][[0, -1], 0, 0] == pytest.approx([15, 19.96])
    assert out["timing"]["event_duration"] == pytest.approx(10)
    assert out["pre_pose"].shape == (32, 3, 3)
    assert out["pre_pose"][..., 2].max() == 1  # confidence clipped to [0, 1]


def test_resample_does_not_bridge_gaps():
    features = np.ones((100, 1, 3), np.float32)
    valid = np.ones((100, 1), bool)
    valid[40:60] = False
    features[40:60] = 0
    out = resample(features, valid, steps=100)
    # Point 59 brackets valid frame 60, so it takes that neighbour.
    assert (out[40:59] == 0).all() and (out[59:, :, 2] == 1).all()


@pytest.mark.parametrize("seconds", [10, 8])
def test_reject_clips_without_event_frames(tmp_path, seconds):
    write_clip(tmp_path / "c.npz", seconds)
    with pytest.raises(ValueError, match="no event frames"):
        convert_clip(tmp_path / "c.npz", steps=32)


def test_reject_unobserved_region(tmp_path):
    valid = np.ones((500, 3), bool)
    valid[:125] = False
    write_clip(tmp_path / "c.npz", 20, valid=valid)
    with pytest.raises(ValueError, match="pre_pose has no observed keypoints"):
        convert_clip(tmp_path / "c.npz", steps=32)


def make_dataset(root):
    clips = [
        (f"{i}_v_ArmFlapping_{i:02d}_seg1_ctx", split, seconds)
        for i, split, seconds in [
            (1, "train", 20),
            (2, "train", 20),
            (3, "train", 20),
            (4, "test", 20),
            (5, "test", 10),
            (6, "test", 20),
        ]
    ]
    for split in ("train", "test"):
        (root / "ViTPoseClean" / split).mkdir(parents=True)
        with (root / "ViTPoseClean" / split / "clean_qc.csv").open(
            "w", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["clip", "ok_head", "ok_arms"])
            writer.writerows([c, "True", "True"] for c, s, _ in clips if s == split)
    for clip, split, seconds in clips:
        write_clip(root / "ViTPoseClean" / split / f"{clip}.npz", seconds)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Final_detailed"
    sheet.append(
        [
            "clip_name",
            "Category",
            "FolderName",
            *INDICATORS,
            "valence",
            "arousal",
            "needs_review",
            *[f"{t}__{a}" for t in ("valence", "arousal") for a in "ABC"],
        ]
    )
    # Valence votes positive, neutral, unknown; arousal high, low, low.
    votes = ["1:5", "0.5:4", "2:3", "1:5", "0:4", "0:3"]
    for index, (clip, split, _) in enumerate(clips):
        review = "Y" if index == 5 else None
        sheet.append(
            [clip, "armflapping", split, 1, 0, 0, 0, 0, 0.5, -0.5, review, *votes]
        )
    workbook.save(root / "Final_Annotation_v2.xlsx")
    with (root / TEXT_CSV).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_name", "event_text"])
        writer.writerows([c, f"text {i % 2}"] for i, (c, _, _) in enumerate(clips))


def run(tmp_path, *extra):
    root, out = tmp_path / "dataset", tmp_path / "out"
    make_dataset(root)
    main(
        [
            "--dataset",
            str(root),
            "--output",
            str(out),
            "--val-fraction",
            "0.34",
            "--poses",
            "ViTPoseClean",
            *extra,
        ]
    )
    return out


def test_main_writes_loadable_manifests(tmp_path, capsys):
    out = run(tmp_path, "--text-encoder", "indicators")
    assert "excluded 5_v_ArmFlapping_05_seg1_ctx" in capsys.readouterr().out
    config = ModelConfig(text_dim=5, num_joints=3, joint_features=3)
    datasets = [
        PoseDataset(out / f"{s}.csv", config, s) for s in ("train", "val", "test")
    ]
    assert_group_separation(*datasets)
    assert [len(d) for d in datasets] == [2, 1, 1]
    assert datasets[2].excluded_review == 1
    sample = datasets[2][0]
    assert sample["inputs"]["text_embedding"].tolist() == [1, 0, 0, 0, 0]
    assert sample["target"].tolist() == [0.5, -0.5]
    annotations = ModelConfig(4, 3, 3, model="annotation_film")
    context = PoseDataset(out / "test.csv", annotations, "test")[0]["inputs"]
    assert list(context) == ["context", "pre_pose", "event_pose", "post_pose"]
    assert context["context"].tolist() == [1, 0, 0, 0]  # no_event dropped
    row = datasets[2].rows[0]
    assert (row["valence_votes"], row["arousal_votes"]) == ("1|0|", "1|-1|-1")
    np.testing.assert_allclose(np.load(row["path"])["votes"], [0.5, 1 / 3])


def test_duplicate_videos_share_one_participant():
    assert prepare.video_id("183_v_HandAction_05_0_seg2_ctx") == "109_v_ArmFlapping_59"
    assert prepare.video_id("7_v_Spinning_01_seg3_ctx") == "7_v_Spinning_01"


def test_main_uses_event_text_embeddings(tmp_path, monkeypatch):
    def fake_encoder(texts):
        return {t: np.full(4, float(t[-1]), np.float32) for t in texts}, "fake@1"

    monkeypatch.setattr(prepare, "encode_clip_text", fake_encoder)
    out = run(tmp_path)
    config = ModelConfig(text_dim=4, num_joints=3, joint_features=3)
    test = PoseDataset(out / "test.csv", config, "test")
    # Clip 4 (index 3) carries "text 1".
    assert test[0]["inputs"]["text_embedding"].tolist() == [1, 1, 1, 1]
    assert test.rows[0]["event_text"] == "text 1"
    assert test.rows[0]["text_encoder"] == "fake@1"


def test_speed_channel_is_in_shoulder_widths_per_second(tmp_path):
    path = tmp_path / "clip.npz"
    write_clip(path, 20.0)  # x advances 1 shoulder width per second
    sample = convert_clip(path, 32, speed=True)
    for key in ("pre_pose", "event_pose", "post_pose"):
        assert sample[key].shape == (32, 3, 4)
        np.testing.assert_allclose(sample[key][..., 3], 1.0, rtol=1e-4)
    assert convert_clip(path, 32)["event_pose"].shape == (32, 3, 3)


def test_speed_is_zero_where_unobserved(tmp_path):
    valid = np.ones((500, 3), bool)
    valid[200:300] = False
    path = tmp_path / "clip.npz"
    write_clip(path, 20.0, valid=valid)
    event = convert_clip(path, 32, speed=True)["event_pose"]  # frames 125-374
    hidden = (event[..., 2] == 0).all(1)
    assert hidden.any() and (event[hidden, :, 3] == 0).all()


def test_face_gate_masks_turned_away_dark_and_blurred_steps(tmp_path):
    steps = 4
    cache = tmp_path / "clip.npz"
    frontal = np.ones((3, steps), bool)
    frontal[0, 0] = False  # turned away
    brightness = np.full((3, steps), 100.0)
    brightness[1, 1] = prepare.MIN_FACE_BRIGHTNESS - 1  # too dark
    sharpness = np.full((3, steps), 100.0)
    sharpness[2, 2] = prepare.MIN_FACE_SHARPNESS - 1  # blurred
    detector = np.ones((3, steps))
    detector[2, 3] = prepare.MIN_FACE_DETECTOR - 0.1  # no face detected
    np.savez(
        cache,
        probs=np.full((3, steps, 7), 0.5, np.float32),
        frontal=frontal,
        brightness=brightness,
        sharpness=sharpness,
        detector=detector,
    )
    regions = prepare.face_regions(cache, "probs", steps)
    masks = np.stack(
        [regions[k][:, -1] for k in ("pre_face", "event_face", "post_face")]
    )
    assert (
        masks.tolist()
        == (frontal & (brightness > 50) & (sharpness > 50) & (detector == 1)).tolist()
    )
    assert (regions["pre_face"][0, :-1] == 0).all()
    assert (regions["pre_face"][1, :-1] == 0.5).all()
    empty = prepare.face_regions(tmp_path / "missing.npz", "embedding", steps)
    assert empty["event_face"].shape == (steps, 769) and not empty["event_face"].any()


def test_single_annotator_sheet_uses_values_and_drops_conflicts(tmp_path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(
        ["clip_name", "Category", "FolderName"]
        + [f"{prefix}[0/1]: ConfScore[1-5]" for prefix in SINGLE_ANNOTATOR_COLUMNS]
    )
    # Columns: valence, arousal, social, auditory, object, visual, no event.
    sheet.append(
        ["a", "spinning", "train", "0.5:3", "1:2", "1:5", "0:4", "0:5", "1:4", "0:5"]
    )
    sheet.append(
        ["b", "spinning", "train", "1:3", "0:2", "1:5", "0:4", "0:5", "0:4", "1:5"]
    )
    sheet.append(
        ["c", "spinning", "test", "2:3", "0:2", "0:5", "0:4", "0:5", "0:4", "1:5"]
    )
    sheet.append(
        ["d", "spinning", "test", "0:3", "0:2", "0:5", "0:4", "0:5", "0:4", "1:5"]
    )
    workbook.save(tmp_path / "sheet.xlsx")
    records, excluded = load_single_annotator(tmp_path / "sheet.xlsx", "Olivia")
    assert sorted(records) == ["a", "d"]
    assert dict(excluded) == {
        "b": "no_event is 1 alongside another indicator",
        "c": "unknown valence (2)",
    }
    a = records["a"]
    assert [a[k] for k in INDICATORS] == [1, 0, 0, 1, 0]
    assert (a["valence"], a["arousal"]) == (0.0, 1.0)  # neutral -> not positive
    assert (a["valence__Olivia"], a["arousal__Olivia"]) == ("0.5:3", "1:2")
    assert records["d"]["valence"] == -1.0
