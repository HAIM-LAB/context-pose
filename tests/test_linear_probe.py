import numpy as np
import pytest

from asd_contextpose.linear_probe import fit_predict, region_features


def moving_clip(frames=50, fps=25.0):
    """Both wrists move right at 2 shoulder widths/s; everything is observed."""
    times = np.arange(frames) / fps
    norm = np.zeros((frames, 11, 2), np.float32)
    norm[:, [9, 10], 0] = 2.0 * times[:, None]
    norm[:, [9, 10], 1] = 1.0  # wrists one shoulder width below the shoulders
    return {
        "frame_time_s": times,
        "keypoints_norm": norm,
        "valid": np.ones((frames, 11), bool),
        "root_px": np.zeros((frames, 2), np.float32),
        "scale_px": np.float32(100.0),
    }


def test_region_features_measure_speed_in_shoulder_widths_per_second():
    features = region_features(moving_clip(), 0.0, 2.0)
    assert features["coverage"] == 1.0
    assert features["wrist_speed"] == pytest.approx(2.0, rel=1e-4)
    assert features["head_speed"] == 0.0 and features["body_speed"] == 0.0
    assert features["wrist_height"] == pytest.approx(1.0)


def test_region_features_are_nan_when_unobserved():
    clip = moving_clip()
    clip["valid"][:] = False
    features = region_features(clip, 0.0, 2.0)
    assert features["coverage"] == 0.0
    assert np.isnan(features["wrist_speed"]) and np.isnan(features["head_bob"])


def test_intercept_only_fits_mean_for_l2_and_median_for_l1():
    y = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, -1.0], [3.0, 1.0]])
    x = np.zeros((4, 0))
    l2 = fit_predict(x, y, x[:1], alpha=1.0, loss="l2")[0]
    l1 = fit_predict(x, y, x[:1], alpha=1.0, loss="l1")[0]
    assert l2 == pytest.approx(y.mean(0))
    assert l1 == pytest.approx([0.0, 1.0], abs=1e-2)
