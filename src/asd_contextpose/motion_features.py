"""Motion features at the model's pose steps: a motion descriptor and motion-guided VST.

For every clip and region (pre/event/post), a wide window of MOTION_WINDOW
frames at RATE fps (about 2 s) around each of the `steps` resampling points is
cropped to the child as in `video_features`, and described two ways:

- `motion`: a hand-designed descriptor of the child's movement. Farneback flow
  on a grey HOG_SIZE crop, with each frame pair's median flow (camera motion and
  whole-body drift) removed, gives per-cell motion energy (GRID x GRID), a
  time-exposure image (per-pixel temporal standard deviation of the grey crop,
  per cell), statistics of the motion-energy time course, its periodicity
  (spectral band shares of the energy, and the dominant frequency and its
  strength for the horizontal and vertical flow), the flow's direction
  (horizontal / vertical energy), its rotation and expansion (mean curl,
  |curl| and divergence), and the removed camera motion.
- `vst_motion`: the frozen Video Swin Transformer of `video_features` on
  WINDOW frames sampled from the wide window by motion (after MGSampler, Zhi et
  al. 2021): at equal steps of the cumulative motion energy, so fast movement
  gets more frames and still stretches fewer, in temporal order.

Writes one cache NPZ per clip, in the layout `add_streams` reads:

    motion      [3, steps, MOTION_DIM]  float32
    vst_motion  [3, steps, D]           float16
    tracked     [3, steps]              the shoulder midpoint is tracked at the step
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from .face import step_frames
from .prepare import FLANK_S, load_annotations
from .video_features import (
    BACKBONES,
    CHILD_DROP,
    CHILD_SIDE,
    GRID,
    HOG_SIZE,
    RATE,
    WINDOW,
    VideoEncoder,
    read_windows,
    root_track,
)

MOTION_WINDOW = 2 * WINDOW  # frames in the wide window: about 2 s at RATE
BANDS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.5, RATE / 2)  # Hz edges of the energy bands
MOTION_DIM = 2 * GRID * GRID + 4 + (len(BANDS) - 1) + 4 + 3 + 3 + 1


def wide_frames(centre: int, fps_in: float, last: int) -> np.ndarray:
    """MOTION_WINDOW video-frame indices at about RATE fps, centred on `centre`."""
    stride = max(1, round(fps_in / RATE))
    offsets = stride * (np.arange(MOTION_WINDOW) - MOTION_WINDOW // 2)
    return np.clip(centre + offsets, 0, last)


def cell_means(image: np.ndarray) -> np.ndarray:
    """[GRID * GRID] mean of a square image over each cell."""
    size = image.shape[0]
    return image.reshape(GRID, size // GRID, GRID, size // GRID).mean((1, 3)).ravel()


def dominant(signal: np.ndarray) -> list[float]:
    """Dominant frequency (as a share of Nyquist) and its share of the power."""
    power = np.abs(np.fft.rfft(signal - signal.mean()))[1:] ** 2
    if power.sum() <= 1e-12:
        return [0.0, 0.0]
    peak = int(power.argmax())
    hertz = np.fft.rfftfreq(len(signal), 1 / RATE)[1:]
    return [hertz[peak] / (RATE / 2), float(power[peak] / power.sum())]


def motion_flow(window: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Grey frames, relative flow [T-1, S, S, 2] and median flow [T-1, 2]."""
    grey = [
        cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
            (HOG_SIZE, HOG_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        for frame in window
    ]
    flow = np.stack(
        [
            cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            for a, b in zip(grey, grey[1:])
        ]
    )
    shift = np.median(flow.reshape(len(flow), -1, 2), 1)
    return grey, flow - shift[:, None, None], shift


def motion_descriptor(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[MOTION_DIM] descriptor and [T-1] motion energy per frame pair."""
    grey, relative, shift = motion_flow(window)
    vx, vy = relative[..., 0], relative[..., 1]
    magnitude = np.hypot(vx, vy)
    energy = magnitude.mean((1, 2))  # per frame pair
    exposure = np.stack(grey).astype(np.float32).std(0) / 255
    peaks = (energy[1:-1] > energy[:-2]) & (energy[1:-1] > energy[2:])
    course = [
        np.log1p(energy.mean()),
        np.log1p(energy.std()),
        np.log1p(energy.max()),
        peaks.mean(),
    ]
    power = np.abs(np.fft.rfft(energy - energy.mean())) ** 2
    hertz = np.fft.rfftfreq(len(energy), 1 / RATE)
    bands = np.array(
        [power[(hertz >= lo) & (hertz < hi)].sum() for lo, hi in zip(BANDS, BANDS[1:])]
    )
    bands = bands / max(bands.sum(), 1e-12)
    periodic = dominant(vx.mean((1, 2))) + dominant(vy.mean((1, 2)))
    ax, ay = np.abs(vx).mean(), np.abs(vy).mean()
    direction = [np.log1p(ax), np.log1p(ay), ay / max(ax + ay, 1e-8)]
    curl = np.gradient(vy, axis=2) - np.gradient(vx, axis=1)
    divergence = np.gradient(vx, axis=2) + np.gradient(vy, axis=1)
    rotation = [curl.mean(), np.abs(curl).mean(), divergence.mean()]
    descriptor = np.concatenate(
        [
            np.log1p(cell_means(magnitude.mean(0))),
            cell_means(exposure),
            course,
            bands,
            periodic,
            direction,
            rotation,
            [np.log1p(np.hypot(*shift.T).mean())],
        ]
    )
    return descriptor.astype(np.float32), energy


def motion_slots(energy: np.ndarray) -> np.ndarray:
    """WINDOW frame positions in the wide window, at equal steps of cumulative motion.

    Frame i's motion is the energy of the pair ending at it (frame 0 takes the
    first pair's); a small floor keeps a still window uniformly sampled.
    """
    motion = np.concatenate([energy[:1], energy]) + 1e-3 * max(energy.mean(), 1e-6)
    cumulative = np.cumsum(motion) / motion.sum()
    quantiles = (np.arange(WINDOW) + 0.5) / WINDOW
    return np.searchsorted(cumulative, quantiles).clip(0, len(motion) - 1)


def extract_clip(pose_npz: Path, video: Path, encoder: VideoEncoder, steps: int):
    with np.load(pose_npz, allow_pickle=False) as data:
        root, times = data["root_px"], data["frame_time_s"]
        scale = float(data["scale_px"])
        fps, fps_in = float(data["fps"]), float(data["fps_in"])
        last = int(data["n_frames_in"]) - 1
    frames = step_frames(len(root), fps, steps)
    track = root_track(root)
    needs: dict[int, list[tuple]] = {}
    for window, (region, step) in enumerate(np.ndindex(frames.shape)):
        frame = frames[region, step]
        centre = (
            None if track is None else track[frame] + np.array([0, CHILD_DROP * scale])
        )
        middle = min(int(round(times[frame] * fps_in)), last)
        for slot, index in enumerate(wide_frames(middle, fps_in, last)):
            needs.setdefault(int(index), []).append(
                (window, slot, centre, CHILD_SIDE * scale)
            )
    wide = read_windows(video, needs, frames.size, MOTION_WINDOW)
    descriptors, sampled = [], []
    for window in wide:
        descriptor, energy = motion_descriptor(window)
        descriptors.append(descriptor)
        sampled.append(window[motion_slots(energy)])
    return {
        "motion": np.stack(descriptors).reshape(3, steps, -1),
        "vst_motion": encoder(np.stack(sampled))
        .reshape(3, steps, -1)
        .astype(np.float16),
        "tracked": np.isfinite(root[frames]).all(-1),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--poses", default="ViTPoseFilled")
    parser.add_argument("--output", type=Path, default=Path("dataset/MotionFeatures"))
    parser.add_argument("--backbone", choices=BACKBONES, default="swin3d_b")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    encoder = VideoEncoder(args.backbone, args.device)
    annotations = load_annotations(args.dataset / "Final_Annotation_v2.xlsx")
    for index, (clip, ann) in enumerate(sorted(annotations.items()), 1):
        split = ann["FolderName"]
        target = args.output / split / f"{clip}.npz"
        pose = args.dataset / args.poses / split / f"{clip}.npz"
        video = args.dataset / "full_video" / split / f"{clip}.mp4"
        if target.exists() or not pose.exists() or not video.exists():
            continue
        with np.load(pose, allow_pickle=False) as data:
            if len(data["valid"]) <= 2 * round(FLANK_S * float(data["fps"])):
                continue  # no event frames; prepare excludes these clips too
        features = extract_clip(pose, video, encoder, args.steps)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target, **features, backbone=f"{args.backbone}:{BACKBONES[args.backbone]}"
        )
        print(
            f"{index}/{len(annotations)} {clip}: tracked {features['tracked'].mean():.2f}"
        )


if __name__ == "__main__":
    main()
