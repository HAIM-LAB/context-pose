import numpy as np

from asd_contextpose.face import (
    CROP,
    DETECT_SIZE,
    detector_score,
    face_centre,
    square_crop,
    step_frames,
)


def test_step_frames_follow_prepare_regions():
    frames = step_frames(n_frames=500, fps=25.0, steps=32)
    assert frames.shape == (3, 32)
    assert frames[0, 0] == 0 and frames[0, -1] == 124  # 5 s pre-event
    assert frames[1, 0] == 125 and frames[1, -1] == 374
    assert frames[2, 0] == 375 and frames[2, -1] == 499


def test_face_centre_uses_nearby_frames_and_needs_two_face_points():
    px = np.zeros((5, 11, 2), np.float32)
    px[:, :5] = [[10, 20], [12, 18], [8, 18], [14, 20], [6, 20]]
    valid = np.zeros((5, 11), bool)
    valid[3, [0, 1]] = True
    np.testing.assert_allclose(face_centre(px, valid, 2), [11, 19])
    valid[3] = False
    valid[2, 3] = True  # one ear only
    assert face_centre(px, valid, 2) is None


def test_square_crop_pads_outside_the_frame():
    frame = np.full((100, 50, 3), 255, np.uint8)
    crop = square_crop(frame, np.array([0.0, 0.0]), side=40)
    assert crop.shape == (CROP, CROP, 3)
    assert crop[: CROP // 2 - 2, : CROP // 2 - 2].max() == 0  # outside: zero
    assert crop[CROP // 2 + 2 :, CROP // 2 + 2 :].min() == 255  # inside: frame


class FakeDetector:
    """Returns YuNet-format rows: x, y, w, h, 10 landmark values, score."""

    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, image):
        assert image.shape == (DETECT_SIZE, DETECT_SIZE, 3)
        rows = [[x, y, w, h, *[0] * 10, score] for x, y, w, h, score in self.boxes]
        return 1, np.array(rows, np.float32) if rows else None


def test_detector_score_counts_only_faces_at_the_tracked_centre():
    frame = np.zeros((400, 400, 3), np.uint8)
    centre, middle = np.array([200.0, 200.0]), DETECT_SIZE / 2
    at_centre = (middle - 20, middle - 20, 40, 40, 0.9)
    elsewhere = (0, 0, 30, 30, 0.99)
    assert detector_score(FakeDetector([elsewhere]), frame, centre, 100) == 0.0
    score = detector_score(FakeDetector([elsewhere, at_centre]), frame, centre, 100)
    assert score == np.float32(0.9)
    assert detector_score(FakeDetector([]), frame, centre, 100) == 0.0
