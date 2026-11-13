from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from asd_contextpose.wholebody import (
    FLANK_FRAMES,
    INPUT_SIZE,
    WholeBodyViTPose,
    box_transform,
    build_clip,
    decode,
    follow,
    interpolate_gaps,
)

JOINTS = 133


def test_model_maps_input_to_quarter_resolution_heatmaps():
    model = WholeBodyViTPose(dim=32, depth=1, heads=4).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, INPUT_SIZE[1], INPUT_SIZE[0]))
    assert out.shape == (2, JOINTS, 64, 48)


def test_decode_returns_peak_in_image_pixels():
    box = np.array([100.0, 200.0, 196.0, 328.0])  # 96 x 128: already 3:4
    matrix = box_transform(box)
    heatmaps = np.zeros((1, 2, 64, 48), np.float32)
    heatmaps[0, 0, 32, 24] = 0.9  # centre of the input
    heatmaps[0, 1, 0, 0] = 0.5
    points, peaks = decode(heatmaps, [matrix])
    assert points[0, 0] == pytest.approx([148.0, 264.0], abs=1e-3)
    # The padded box (x1.25) starts 12 px left and 16 px above the box.
    assert points[0, 1] == pytest.approx([88.0, 184.0], abs=1e-3)
    assert peaks[0].tolist() == pytest.approx([0.9, 0.5])


def person(x: float):
    pose = np.tile(np.array([[x, 100.0]]), (JOINTS, 1))
    pose[:17, 1] += np.arange(17)
    return pose, np.ones(JOINTS, np.float32)


def test_follow_keeps_the_nearest_person_and_stops_after_misses():
    start = (person(0)[0], np.ones(JOINTS, bool))
    frames = {
        0: [person(2), person(80)],  # child drifts 2 px per frame; scale 10
        1: [person(80)],
        2: [person(4), person(80)],
        5: [person(4)],
    }
    chosen = follow(range(6), frames, start, lambda i: None, scale=10, max_misses=1)
    assert sorted(chosen) == [0, 2]
    assert chosen[2][0][0, 0] == 4


def test_follow_prefers_the_old_track_as_reference():
    start = (person(0)[0], np.ones(JOINTS, bool))
    frames = {0: [person(0), person(80)]}
    old = (person(80)[0][:11], np.ones(11, bool))
    chosen = follow([0], frames, start, lambda i: old, scale=10, max_misses=1)
    assert chosen[0][0][0, 0] == 80


def test_interpolate_gaps_fills_short_gaps_only():
    px = np.zeros((10, 1, 2), np.float32)
    px[:, 0, 0] = np.arange(10)
    conf = np.ones((10, 1), np.float32)
    valid = np.ones((10, 1), bool)
    valid[2:4] = False  # 2 frames
    valid[5:9] = False  # 4 frames
    px[~valid] = np.nan
    filled = interpolate_gaps(px, conf, valid, np.arange(10), max_gap=2)
    assert filled[:, 0].tolist() == [i in (2, 3) for i in range(10)]
    assert px[2:4, 0, 0].tolist() == pytest.approx([2, 3])
    assert not valid[5:9].any()


class FakeSearch:
    """One person per frame, at the event's pose plus a small offset."""

    def __init__(self, pose):
        self.pose = pose

    def candidates(self, bgr):
        conf = np.full(JOINTS, 0.8, np.float32)
        return [(self.pose + 1.0, conf), (self.pose + 500.0, conf)]


def write_video(path, frames: int):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (32, 24))
    for _ in range(frames):
        writer.write(np.zeros((24, 32, 3), np.uint8))
    writer.release()


def test_build_clip_puts_event_after_five_seconds_and_fills_flanks(tmp_path):
    length, event_frames = FLANK_FRAMES * 2 + 20, 20
    pose = np.tile(np.arange(JOINTS, dtype=np.float32)[:, None], (1, 2)) * 3 + 50
    pose[5], pose[6] = (40, 60), (60, 60)  # shoulders 20 px apart
    event = {
        "keypoints_px": np.tile(pose, (event_frames, 1, 1)),
        "kp_conf": np.full((event_frames, JOINTS), 0.9, np.float32),
        "valid": np.ones((event_frames, JOINTS), bool),
        "interpolated": np.zeros((event_frames, JOINTS), bool),
        "root_px": np.tile([50.0, 60.0], (event_frames, 1)).astype(np.float32),
        "scale_px": np.float32(20),
        "scale_source": np.array("shoulders"),
        "joint_names": np.array([f"j{i}" for i in range(JOINTS)]),
        "width": np.int64(32),
        "height": np.int64(24),
        "layout": np.array("coco_wholebody_133"),
        "pose_model": np.array("test"),
    }
    event["keypoints_norm"] = ((event["keypoints_px"] - [50.0, 60.0]) / 20).astype(
        np.float32
    )
    old = {
        "keypoints_px": np.tile(pose[:11], (length, 1, 1)),
        "valid": np.ones((length, 11), bool),
        "fps_in": np.float32(25),
        "n_frames_in": np.int64(length),
    }
    np.savez(tmp_path / "event.npz", **event)
    np.savez(tmp_path / "old.npz", **old)
    write_video(tmp_path / "clip.mp4", length)
    args = SimpleNamespace(max_misses=12, max_gap=12)
    data = build_clip(
        tmp_path / "event.npz",
        tmp_path / "old.npz",
        tmp_path / "clip.mp4",
        FakeSearch(pose),
        args,
    )
    span = slice(FLANK_FRAMES, FLANK_FRAMES + event_frames)
    assert data["from_event"].sum() == event_frames and data["from_event"][span].all()
    assert np.array_equal(data["keypoints_norm"][span], event["keypoints_norm"])
    assert data["valid"].all() and bool(data["old_track_trusted"])
    flank = data["keypoints_px"][0]
    assert flank == pytest.approx(pose + 1.0)  # the near person, not the far one
    assert data["keypoints_norm"][0] == pytest.approx((pose + 1.0 - [51, 61]) / 20)
