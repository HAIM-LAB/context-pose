"""Frozen Video Swin Transformer and HOG-HOF features at the model's pose steps.

For every clip and region (pre/event/post), a short window of video around
each of the `steps` resampling points `prepare` uses is cropped to the child
and described two ways:

- `vst`: a Kinetics-400 Video Swin Transformer (torchvision Swin3D), frozen,
  with its last-stage tokens averaged over space and time (the classifier's
  input, without the classifier).
- `hoghof`: hand-designed appearance and motion descriptors after Konecny and
  Hagara (2013): a histogram of oriented gradients of the window's middle
  frame, a histogram of optical flow over the window with the crop's median
  flow (camera motion and whole-body drift) removed, and two log motion
  energies (the relative flow and the removed median flow).

The window is WINDOW frames sampled at RATE fps, centred on the step. The crop
is a square of CHILD_SIDE shoulder widths centred CHILD_DROP shoulder widths
below the tracked shoulder midpoint (`root_px`), which is interpolated across
untracked frames and median-smoothed. A clip with no tracked frame at all is
cropped to the full frame. Writes one cache NPZ per clip, in the layout of the
EmotionCLIP features:

    vst     [3, steps, D]            float16 (D = 768 for swin3d_t/s, 1024 for b)
    hoghof  [3, steps, HOGHOF_DIM]   float32
    tracked [3, steps]               the shoulder midpoint is tracked at the step
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from .face import step_frames
from .prepare import FLANK_S, load_annotations

WINDOW, RATE = 16, 15.0  # frames per window and their rate: about 1 s
CHILD_SIDE, CHILD_DROP = 4.5, 0.9  # shoulder widths (checked visually)
SMOOTH = 9  # pose frames in the root's moving median
CROP = 224
BACKBONES = {  # torchvision Swin3D and its Kinetics-400 weights
    "swin3d_t": "KINETICS400_V1",
    "swin3d_s": "KINETICS400_V1",
    "swin3d_b": "KINETICS400_IMAGENET22K_V1",
}
KINETICS_MEAN, KINETICS_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
HOG_SIZE, GRID, HOG_BINS, HOF_BINS = 96, 4, 9, 8
HOGHOF_DIM = GRID * GRID * (HOG_BINS + HOF_BINS) + 2


def root_track(root: np.ndarray) -> np.ndarray | None:
    """[T, 2] root interpolated over untracked frames and median-smoothed."""
    tracked = np.isfinite(root).all(1)
    if not tracked.any():
        return None
    index = np.arange(len(root))
    filled = np.stack(
        [np.interp(index, index[tracked], root[tracked, axis]) for axis in (0, 1)], 1
    )
    half = SMOOTH // 2
    padded = np.pad(filled, ((half, half), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, SMOOTH, axis=0)
    return np.median(windows, -1)


def window_frames(centre: int, fps_in: float, last: int) -> np.ndarray:
    """WINDOW video-frame indices at about RATE fps, centred on `centre`."""
    stride = max(1, round(fps_in / RATE))
    offsets = stride * (np.arange(WINDOW) - WINDOW // 2)
    return np.clip(centre + offsets, 0, last)


def crop(frame: np.ndarray, centre: np.ndarray | None, side: float) -> np.ndarray:
    """RGB CROP-wide square around `centre` (the full frame if None), zero-padded."""
    height, width = frame.shape[:2]
    if centre is None:
        centre, side = np.array([width / 2, height / 2]), max(width, height)
    size = max(int(round(side)), 1)
    x0, y0 = np.round(centre - side / 2).astype(int)
    out = np.zeros((size, size, 3), np.uint8)
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + size, width), min(y0 + size, height)
    if xb > xa and yb > ya:
        out[ya - y0 : yb - y0, xa - x0 : xb - x0] = frame[ya:yb, xa:xb]
    out = cv2.resize(out, (CROP, CROP), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def cell_histograms(
    magnitude: np.ndarray, angle: np.ndarray, bins: int, period: float
) -> np.ndarray:
    """[GRID * GRID * bins] magnitude-weighted orientation histograms per cell."""
    size = magnitude.shape[0]
    cell = np.arange(size) * GRID // size
    cells = cell[:, None] * GRID + cell[None, :]
    which = np.floor(angle / period * bins).astype(int) % bins
    return np.bincount(
        (cells * bins + which).ravel(),
        weights=magnitude.ravel(),
        minlength=GRID * GRID * bins,
    )


def hellinger(histogram: np.ndarray) -> np.ndarray:
    return np.sqrt(histogram / max(histogram.sum(), 1e-8))


def hog_hof(window: np.ndarray) -> np.ndarray:
    """[HOGHOF_DIM] descriptor of a [WINDOW, CROP, CROP, 3] RGB window."""
    grey = [
        cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
            (HOG_SIZE, HOG_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        for frame in window
    ]
    middle = grey[len(grey) // 2].astype(np.float32)
    gx = cv2.Sobel(middle, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(middle, cv2.CV_32F, 0, 1)
    hog = cell_histograms(np.hypot(gx, gy), np.arctan2(gy, gx) % np.pi, HOG_BINS, np.pi)
    flow = np.stack(
        [
            cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            for a, b in zip(grey, grey[1:])
        ]
    )  # [WINDOW - 1, HOG_SIZE, HOG_SIZE, 2] in crop pixels per frame pair
    shift = np.median(flow.reshape(len(flow), -1, 2), 1)
    relative = flow - shift[:, None, None]
    magnitude = np.hypot(relative[..., 0], relative[..., 1])
    angle = np.arctan2(relative[..., 1], relative[..., 0]) % (2 * np.pi)
    hof = sum(
        cell_histograms(m, a, HOF_BINS, 2 * np.pi) for m, a in zip(magnitude, angle)
    )
    energy = np.log1p([magnitude.mean(), np.hypot(*shift.T).mean()])
    return np.concatenate([hellinger(hog), hellinger(hof), energy]).astype(np.float32)


def read_windows(
    video: Path, needs: dict[int, list[tuple]], count: int, length: int = WINDOW
) -> np.ndarray:
    """[count, length, CROP, CROP, 3] crops; `needs` maps a frame to its slots.

    Decodes sequentially (seeking is unreliable). Slots past the last decodable
    frame repeat the window's nearest decoded frame.
    """
    windows = np.zeros((count, length, CROP, CROP, 3), np.uint8)
    filled = np.zeros((count, length), bool)
    capture, index, stop = cv2.VideoCapture(str(video)), 0, max(needs)
    while index <= stop:
        ok, frame = capture.read()
        if not ok:
            break
        for window, slot, centre, side in needs.get(index, ()):
            windows[window, slot] = crop(frame, centre, side)
            filled[window, slot] = True
        index += 1
    capture.release()
    for window in np.flatnonzero(~filled.all(1) & filled.any(1)):
        have = np.flatnonzero(filled[window])
        for slot in np.flatnonzero(~filled[window]):
            windows[window, slot] = windows[window, have[np.abs(have - slot).argmin()]]
    return windows


class VideoEncoder:
    def __init__(self, backbone: str, device: str):
        import torch
        from torchvision.models import get_model

        self.torch, self.device = torch, torch.device(device)
        model = get_model(backbone, weights=BACKBONES[backbone])
        self.model = model.eval().to(self.device)
        self.dim = model.head.in_features
        self.mean = torch.tensor(KINETICS_MEAN, device=self.device)[:, None, None, None]
        self.std = torch.tensor(KINETICS_STD, device=self.device)[:, None, None, None]

    def __call__(self, windows: np.ndarray, batch: int = 8) -> np.ndarray:
        """[N, D] features of [N, WINDOW, CROP, CROP, 3] uint8 RGB windows."""
        torch, m, out = self.torch, self.model, []
        with torch.no_grad():
            for start in range(0, len(windows), batch):
                x = torch.from_numpy(windows[start : start + batch]).to(self.device)
                x = x.permute(0, 4, 1, 2, 3).float() / 255  # B C T H W
                x = (x - self.mean) / self.std
                tokens = m.norm(m.features(m.pos_drop(m.patch_embed(x))))
                out.append(tokens.mean((1, 2, 3)).cpu())  # B _T _H _W C -> B C
        return torch.cat(out).numpy()


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
        for slot, index in enumerate(window_frames(middle, fps_in, last)):
            needs.setdefault(int(index), []).append(
                (window, slot, centre, CHILD_SIDE * scale)
            )
    windows = read_windows(video, needs, frames.size)
    return {
        "vst": encoder(windows).reshape(3, steps, -1).astype(np.float16),
        "hoghof": np.stack([hog_hof(w) for w in windows]).reshape(3, steps, -1),
        "tracked": np.isfinite(root[frames]).all(-1),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--poses", default="ViTPoseFilled")
    parser.add_argument("--output", type=Path, default=Path("dataset/VideoFeatures"))
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
