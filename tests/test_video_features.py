import numpy as np
import pytest

from asd_contextpose import add_streams
from asd_contextpose import video_features as vf


def test_root_track_interpolates_smooths_and_handles_no_track():
    root = np.full((40, 2), np.nan, np.float32)
    root[5], root[35] = (100, 200), (400, 200)
    track = vf.root_track(root)
    assert track.shape == (40, 2) and np.isfinite(track).all()
    assert np.allclose(track[:5], (100, 200))  # held before the first tracked frame
    assert np.allclose(track[20], (250, 200), atol=1)  # interpolated between them
    assert vf.root_track(np.full((10, 2), np.nan)) is None


def test_window_frames_are_centred_at_the_rate_and_clipped():
    frames = vf.window_frames(100, 30.0, 500)
    assert len(frames) == vf.WINDOW and frames[vf.WINDOW // 2] == 100
    assert np.all(np.diff(frames) == 2)  # 30 fps video sampled at 15 fps
    edge = vf.window_frames(3, 30.0, 10)
    assert edge.min() == 0 and edge.max() <= 10


def test_crop_pads_outside_the_frame_and_falls_back_to_the_full_frame():
    frame = np.full((100, 200, 3), 255, np.uint8)
    corner = vf.crop(frame, np.array([0.0, 0.0]), 50)
    assert corner.shape == (vf.CROP, vf.CROP, 3)
    assert corner[0, 0].sum() == 0 and corner[-1, -1].sum() > 0
    full = vf.crop(frame, None, 0)
    assert full[vf.CROP // 2, vf.CROP // 2].sum() > 0 and full[0, 0].sum() == 0


def test_hog_hof_sees_motion_but_not_camera_drift():
    rng = np.random.default_rng(0)
    texture = (rng.random((vf.CROP + 64, vf.CROP + 64)) * 255).astype(np.uint8)
    texture = np.repeat(texture[..., None], 3, -1)
    still = np.stack([texture[: vf.CROP, : vf.CROP]] * vf.WINDOW)
    drift = np.stack(
        [texture[: vf.CROP, 2 * t : 2 * t + vf.CROP] for t in range(vf.WINDOW)]
    )
    patch = still.copy()
    for t in range(vf.WINDOW):  # one block moves on a still background
        patch[t, 60:120, 60 + 3 * t : 120 + 3 * t] = texture[:60, :60]
    out = {
        name: vf.hog_hof(w)
        for name, w in (("still", still), ("drift", drift), ("patch", patch))
    }
    for descriptor in out.values():
        assert descriptor.shape == (vf.HOGHOF_DIM,) and np.isfinite(descriptor).all()
    relative, removed = -2, -1  # log motion energies
    assert out["still"][relative] < 0.05 and out["still"][removed] < 0.05
    assert (
        out["drift"][removed] > 0.3 and out["drift"][relative] < out["drift"][removed]
    )
    assert out["patch"][relative] > out["still"][relative] + 0.05


def test_stream_arrays_append_the_tracked_flag(tmp_path):
    cache = tmp_path / "clip.npz"
    tracked = np.zeros((3, 4), bool)
    tracked[1, 2] = True
    np.savez(cache, vst=np.ones((3, 4, 5), np.float16), tracked=tracked)
    arrays = add_streams.stream_arrays(cache, ["vst"], 4)
    assert sorted(arrays) == ["event_vst", "post_vst", "pre_vst"]
    assert (
        arrays["event_vst"].shape == (4, 6) and arrays["event_vst"].dtype == np.float32
    )
    assert arrays["event_vst"][2, -1] == 1 and arrays["pre_vst"][:, -1].sum() == 0
    with pytest.raises(ValueError, match="has shape"):
        add_streams.stream_arrays(cache, ["vst"], 8)
