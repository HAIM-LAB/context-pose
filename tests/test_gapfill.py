import cv2
import numpy as np
import pytest

from asd_contextpose.gapfill import (
    ROTATIONS,
    look_distance,
    match_distance,
    torso_hist,
    unrotate,
)


@pytest.mark.parametrize("angle", sorted(ROTATIONS))
def test_unrotate_inverts_frame_rotation(angle):
    height, width = 5, 8
    frame = np.zeros((height, width), np.uint8)
    frame[1, 6] = 255  # original pixel at x=6, y=1
    rotated = cv2.rotate(frame, ROTATIONS[angle])
    y, x = np.argwhere(rotated == 255)[0]
    back = unrotate(np.array([[x, y]], float), angle, width, height)
    assert back.tolist() == [[6, 1]]


def test_match_distance_in_shoulder_widths():
    ref = (np.zeros((11, 2)), np.ones(11, bool))
    pose = np.full((11, 2), [3.0, 4.0])
    conf = np.ones(11)
    assert match_distance(pose, conf, ref, scale=10) == pytest.approx(0.5)
    conf[2:] = 0.1  # only two confident shared joints
    assert match_distance(pose, conf, ref, scale=10) == np.inf


def two_tone_person(flip: bool):
    """Red above the shoulder line, blue below; face above (or below if flipped)."""
    image = np.zeros((100, 100, 3), np.uint8)
    image[:50] = (0, 0, 255)  # BGR red
    image[50:] = (255, 0, 0)  # BGR blue
    pose = np.zeros((11, 2))
    pose[5], pose[6] = (30, 50), (70, 50)
    pose[0] = (50, 30)
    if flip:  # upside down: torso above the shoulders
        image = image[::-1].copy()
        pose[:, 1] = 99 - pose[:, 1]
    return image, pose, np.ones(11, bool)


@pytest.mark.parametrize("flip", [False, True])
def test_torso_hist_reads_clothing_below_shoulders(flip):
    image, pose, ok = two_tone_person(flip)
    blue = torso_hist(np.full_like(image, (255, 0, 0)), pose, ok)
    red = torso_hist(np.full_like(image, (0, 0, 255)), pose, ok)
    torso = torso_hist(image, pose, ok)
    assert look_distance(torso, blue) < 0.2 < look_distance(torso, red)


def test_torso_hist_needs_both_shoulders():
    image, pose, ok = two_tone_person(False)
    ok[6] = False
    assert torso_hist(image, pose, ok) is None
