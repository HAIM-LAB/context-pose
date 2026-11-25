import cv2
import numpy as np

from asd_contextpose import motion_features as mf
from asd_contextpose import video_features as vf

ENERGY, CAMERA = 2 * vf.GRID * vf.GRID, -1  # cell energy + time exposure; camera
BANDS = slice(2 * vf.GRID * vf.GRID + 4, 2 * vf.GRID * vf.GRID + 4 + len(mf.BANDS) - 1)
VERTICAL_PERIOD = slice(BANDS.stop + 2, BANDS.stop + 4)  # frequency, strength


def texture(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = (rng.random((vf.CROP + 96, vf.CROP + 96)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(np.repeat(image[..., None], 3, -1), (5, 5), 1.5)


def oscillating(hertz: float) -> np.ndarray:
    """A smooth bright blob moving up and down at `hertz` over a blurred background."""
    rng = np.random.default_rng(0)
    background = cv2.GaussianBlur(
        (rng.random((vf.CROP, vf.CROP)) * 255).astype(np.uint8), (0, 0), 3
    )
    yy, xx = np.mgrid[:50, :50]
    blob = (255 * np.exp(-((yy - 25) ** 2 + (xx - 25) ** 2) / 200)).astype(np.uint8)
    frames = []
    for t in range(mf.MOTION_WINDOW):
        frame = background.copy()
        y = 80 + int(round(8 * np.sin(2 * np.pi * hertz * t / vf.RATE)))
        frame[y : y + 50, 80:130] = np.maximum(frame[y : y + 50, 80:130], blob)
        frames.append(np.repeat(frame[..., None], 3, -1))
    return np.stack(frames)


def test_wide_frames_span_twice_the_window():
    frames = mf.wide_frames(200, 30.0, 1000)
    assert len(frames) == mf.MOTION_WINDOW and frames[mf.MOTION_WINDOW // 2] == 200
    assert np.all(np.diff(frames) == 2)


def test_still_window_has_no_motion_and_is_sampled_uniformly():
    still = np.stack([texture()[: vf.CROP, : vf.CROP]] * mf.MOTION_WINDOW)
    descriptor, energy = mf.motion_descriptor(still)
    assert descriptor.shape == (mf.MOTION_DIM,) and np.isfinite(descriptor).all()
    assert energy.max() < 0.05 and descriptor[CAMERA] < 0.05
    slots = mf.motion_slots(energy)
    assert len(slots) == vf.WINDOW and np.all(np.diff(slots) >= 0)
    assert np.all(np.diff(slots) == 2)  # equal motion: every other frame


def test_oscillation_frequency_and_location_are_found():
    for hertz in (1.5, 2.5, 3.5):
        descriptor, _ = mf.motion_descriptor(oscillating(hertz))
        assert np.isfinite(descriptor).all()
        frequency, strength = descriptor[VERTICAL_PERIOD]
        assert abs(frequency * vf.RATE / 2 - hertz) < 0.3 and strength > 0.5
    cells = descriptor[: ENERGY // 2].reshape(vf.GRID, vf.GRID)
    assert cells[1:3, 1:3].mean() > cells[0, :].mean() + 0.1  # motion sits mid-crop
    assert abs(descriptor[BANDS].sum() - 1) < 1e-5


def test_camera_drift_is_removed_from_the_motion_energy():
    tex = texture()
    drift = np.stack(
        [tex[: vf.CROP, 2 * t : 2 * t + vf.CROP] for t in range(mf.MOTION_WINDOW)]
    )
    descriptor, energy = mf.motion_descriptor(drift)
    assert descriptor[CAMERA] > 0.5 and energy.mean() < 0.3


def test_motion_slots_crowd_where_the_motion_is():
    energy = np.full(mf.MOTION_WINDOW - 1, 0.01)
    energy[20:26] = 5.0
    slots = mf.motion_slots(energy)
    assert len(slots) == vf.WINDOW and np.all(np.diff(slots) >= 0)
    assert np.mean((slots >= 20) & (slots <= 27)) > 0.8
