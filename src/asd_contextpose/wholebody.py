"""133-keypoint context clips: whole-body flanks around the vitpose_133 event poses.

`dataset/vitpose_133` holds COCO-WholeBody poses of the event only. This rebuilds
each `_ctx` clip on the context clip's 25 fps frame grid: the event frames are
copied from vitpose_133 (the event starts 5 s into the context clip), and the
pre-event and post-event frames are estimated from `full_video` with RT-DETR
person boxes and the whole-body ViTPose-H checkpoint (easy_ViTPose,
`torch/wholebody/vitpose-h-wholebody.pth`), so all 133 joints come from the
same pose model family.

The child is followed outward from the event: pre-event frames walk backwards
from the event's first frame, post-event frames forwards from its last. Each
frame accepts the person whose body joints are closest to the reference pose
(in shoulder widths); the reference is the old 11-joint ViTPoseFilled track at
that frame when it agrees with the event track, else the last accepted pose.
Joints follow vitpose_133's conventions: valid at confidence >= 0.3, NaN where
invalid, shoulder-midpoint origin, the event file's shoulder scale; gaps up to
`--max-gap` frames are interpolated within a flank.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from .gapfill import DETECTOR, VALID_CONF

CHECKPOINT = ("JunkyByte/easy_ViTPose", "torch/wholebody/vitpose-h-wholebody.pth")
INPUT_SIZE = (192, 256)  # width, height
BOX_PADDING = 1.25
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
BODY = 17  # COCO body joints lead the whole-body layout
UPPER_BODY = 11  # joints shared with the old ViTPoseFilled track
FPS = 25.0
FLANK_FRAMES = 125  # 5 s at 25 fps: where the event starts in a context clip
AGREE = 1.0  # shoulder widths: old track and event track are the same person


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, dim * 3)
        self.attn.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, dim * 4)
        self.mlp.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.attn.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = nn.functional.scaled_dot_product_attention(q, k, v)
        x = x + self.attn.proj(out.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp.fc2(nn.functional.gelu(self.mlp.fc1(self.norm2(x))))


class WholeBodyViTPose(nn.Module):
    """ViTPose-H (patch 16, 256x192) with the classic two-deconvolution head."""

    def __init__(self, joints: int = 133, dim: int = 1280, depth: int = 32, heads=16):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.patch_embed = nn.Module()
        self.backbone.patch_embed.proj = nn.Conv2d(3, dim, 16, stride=16, padding=2)
        self.backbone.pos_embed = nn.Parameter(torch.zeros(1, 16 * 12 + 1, dim))
        self.backbone.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.backbone.last_norm = nn.LayerNorm(dim, eps=1e-6)
        self.keypoint_head = nn.Module()
        self.keypoint_head.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(dim, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.keypoint_head.final_layer = nn.Conv2d(256, joints, 1)

    def forward(self, images):
        x = self.backbone.patch_embed.proj(images)
        b, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        pos = self.backbone.pos_embed
        x = x + pos[:, 1:] + pos[:, :1]
        for block in self.backbone.blocks:
            x = block(x)
        x = self.backbone.last_norm(x).transpose(1, 2).reshape(b, d, h, w)
        return self.keypoint_head.final_layer(self.keypoint_head.deconv_layers(x))


def box_transform(box: np.ndarray) -> np.ndarray:
    """2x3 affine from an xyxy box (aspect-fixed, padded) to the model input."""
    (x0, y0, x1, y1), (out_w, out_h) = box, INPUT_SIZE
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    if w > h * out_w / out_h:
        h = w * out_h / out_w
    else:
        w = h * out_w / out_h
    w, h = w * BOX_PADDING, h * BOX_PADDING
    src = np.float32([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx, cy]])
    dst = np.float32([[0, 0], [out_w, 0], [out_w / 2, out_h / 2]])
    return cv2.getAffineTransform(src, dst)


def decode(heatmaps: np.ndarray, transforms: list[np.ndarray]):
    """[N, J, H, W] heatmaps -> ([N, J, 2] image pixels, [N, J] peak values).

    Argmax with a quarter-pixel shift toward the higher neighbour.
    """
    n, j, h, w = heatmaps.shape
    flat = heatmaps.reshape(n, j, -1)
    index = flat.argmax(-1)
    peak = np.take_along_axis(flat, index[..., None], -1)[..., 0]
    x, y = (index % w).astype(np.float32), (index // w).astype(np.float32)
    xi, yi = x.astype(int), y.astype(int)
    inside_x, inside_y = (xi > 0) & (xi < w - 1), (yi > 0) & (yi < h - 1)
    ni, ji = np.indices((n, j))
    dx = heatmaps[ni, ji, yi, np.minimum(xi + 1, w - 1)]
    dx = dx - heatmaps[ni, ji, yi, np.maximum(xi - 1, 0)]
    dy = heatmaps[ni, ji, np.minimum(yi + 1, h - 1), xi]
    dy = dy - heatmaps[ni, ji, np.maximum(yi - 1, 0), xi]
    x += np.where(inside_x, np.sign(dx) * 0.25, 0)
    y += np.where(inside_y, np.sign(dy) * 0.25, 0)
    stride = INPUT_SIZE[0] / w
    points = np.stack([x, y], -1) * stride
    out = np.empty_like(points)
    for i, matrix in enumerate(transforms):
        inverse = cv2.invertAffineTransform(matrix)
        out[i] = points[i] @ inverse[:, :2].T + inverse[:, 2]
    return out, peak


class WholeBodySearch:
    """All people in a frame as ([133, 2] pixels, [133] confidence)."""

    def __init__(self, device: str, checkpoint: Path | None = None):
        from huggingface_hub import hf_hub_download
        from transformers import AutoProcessor, RTDetrForObjectDetection

        self.device = device
        self.det_proc = AutoProcessor.from_pretrained(DETECTOR)
        self.det = RTDetrForObjectDetection.from_pretrained(DETECTOR).to(device).eval()
        path = checkpoint or Path(hf_hub_download(*CHECKPOINT))
        # Tensors only: the checkpoint is never unpickled as arbitrary objects.
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        self.pose = WholeBodyViTPose()
        self.pose.load_state_dict(state, strict=True)
        self.pose = self.pose.to(device).eval()

    @torch.no_grad()
    def candidates(self, bgr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        from PIL import Image

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        size = torch.tensor([(image.height, image.width)])
        inputs = self.det_proc(images=image, return_tensors="pt").to(self.device)
        found = self.det_proc.post_process_object_detection(
            self.det(**inputs), target_sizes=size, threshold=VALID_CONF
        )[0]
        boxes = found["boxes"][found["labels"] == 0].cpu().numpy()
        if not len(boxes):
            return []
        transforms = [box_transform(box) for box in boxes]
        crops = np.stack(
            [
                cv2.warpAffine(rgb, m, INPUT_SIZE, flags=cv2.INTER_LINEAR)
                for m in transforms
            ]
        )
        crops = (crops.astype(np.float32) / 255 - MEAN) / STD
        batch = torch.from_numpy(crops).permute(0, 3, 1, 2).to(self.device)
        with torch.autocast(self.device.split(":")[0], dtype=torch.float16):
            heatmaps = self.pose(batch).float().cpu().numpy()
        points, peaks = decode(heatmaps, transforms)
        return list(zip(points, peaks))


def body_distance(pose, conf, ref_pose, ref_ok, scale: float, joints: int) -> float:
    """Mean distance (shoulder widths) over the first `joints` confident in both."""
    shared = (conf[:joints] >= VALID_CONF) & ref_ok[:joints]
    if shared.sum() < 3:
        return np.inf
    gap = np.linalg.norm(pose[:joints][shared] - ref_pose[:joints][shared], axis=1)
    return float(gap.mean() / scale)


def follow(order, frames: dict, start, hint, scale: float, max_misses: int) -> dict:
    """Choose the child in each frame of `order`, walking away from `start`.

    `frames[i]` holds the candidates of frame i; `start` is the event's boundary
    (pose, ok); `hint(i)` gives the old track's (pose, ok) at i, or None.
    Returns {frame: (pose, conf)}.
    """
    ref, misses, chosen = start, 0, {}
    for index in order:
        old = hint(index)
        if old is not None:
            pose_ref, joints, tolerance = old[0], UPPER_BODY, 0.5
            ok_ref = old[1]
        else:
            if misses > max_misses:
                continue
            (pose_ref, ok_ref), joints = ref, BODY
            tolerance = min(0.5 + 0.25 * misses, 2.0)
        scored = [
            (body_distance(p, c, pose_ref, ok_ref, scale, joints), p, c)
            for p, c in frames.get(index, [])
        ]
        best = min(scored, key=lambda s: s[0], default=(np.inf, None, None))
        if best[0] > tolerance:
            misses += 1
            continue
        pose, conf = best[1], best[2]
        ok = conf >= VALID_CONF
        chosen[index] = (pose, conf)
        ref = (np.where(ok[:, None], pose, ref[0]), ref[1] | ok)
        misses = 0
    return chosen


def interpolate_gaps(px, conf, valid, frames: np.ndarray, max_gap: int):
    """Linearly fill per-joint gaps of up to `max_gap` frames inside `frames`."""
    filled = np.zeros_like(valid)
    if not len(frames):
        return filled
    lo, hi = frames[0], frames[-1] + 1
    for joint in range(valid.shape[1]):
        seen = np.flatnonzero(valid[lo:hi, joint]) + lo
        for a, b in zip(seen[:-1], seen[1:]):
            if 1 < b - a <= max_gap + 1:
                t = (np.arange(a + 1, b) - a) / (b - a)
                px[a + 1 : b, joint] = (1 - t)[:, None] * px[a, joint] + t[
                    :, None
                ] * px[b, joint]
                conf[a + 1 : b, joint] = (1 - t) * conf[a, joint] + t * conf[b, joint]
                valid[a + 1 : b, joint] = filled[a + 1 : b, joint] = True
    return filled


def read_frames(video: Path, wanted: set[int], fps_in: float) -> dict[int, np.ndarray]:
    """BGR frames at 25 fps indices `wanted`, decoding the video once in order."""
    source: dict[int, list[int]] = {}
    for index in sorted(wanted):  # below 25 fps, several indices share a frame
        source.setdefault(round(index / FPS * fps_in), []).append(index)
    cap, out, position = cv2.VideoCapture(str(video)), {}, 0
    last = max(source, default=-1)
    while position <= last:
        ok, bgr = cap.read()
        if not ok:
            break
        for index in source.get(position, []):
            out[index] = bgr
        position += 1
    cap.release()
    return out


def build_clip(event_npz: Path, old_npz: Path, video: Path, search, args) -> dict:
    with np.load(event_npz, allow_pickle=False) as source:
        event = dict(source)
    with np.load(old_npz, allow_pickle=False) as source:
        old = dict(source)
    length, joints = len(old["keypoints_px"]), event["keypoints_px"].shape[1]
    scale, fps_in = float(event["scale_px"]), float(old["fps_in"])
    start = FLANK_FRAMES
    stop = min(start + len(event["keypoints_px"]), length)
    px = np.full((length, joints, 2), np.nan, np.float32)
    conf = np.zeros((length, joints), np.float32)
    valid = np.zeros((length, joints), bool)
    interpolated = np.zeros((length, joints), bool)
    span = slice(start, stop)
    px[span] = event["keypoints_px"][: stop - start]
    conf[span] = event["kp_conf"][: stop - start]
    valid[span] = event["valid"][: stop - start]
    interpolated[span] = event["interpolated"][: stop - start]
    from_event = np.zeros(length, bool)
    from_event[span] = True

    # The old track is a usable identity hint only if it follows the same child.
    seen = valid[span, :UPPER_BODY] & old["valid"][span]
    gap = np.linalg.norm(px[span, :UPPER_BODY] - old["keypoints_px"][span], axis=-1)
    trusted = seen.sum() >= 20 and np.median(gap[seen]) / scale <= AGREE

    def hint(index):
        ok = old["valid"][index]
        if not trusted or ok.sum() < 3:
            return None
        return old["keypoints_px"][index], ok

    def boundary(indices):
        for index in indices:
            ok = valid[index, :BODY]
            if ok.sum() >= 3:
                pose = np.where(valid[index][:, None], px[index], 0)
                return pose, valid[index].copy()
        return None

    pre, post = np.arange(0, start), np.arange(stop, length)
    frames = read_frames(video, set(pre) | set(post), fps_in)
    candidates = {i: search.candidates(bgr) for i, bgr in frames.items()}
    for order, anchor in (
        (pre[::-1], boundary(range(start, stop))),
        (post, boundary(range(stop - 1, start - 1, -1))),
    ):
        if anchor is None:
            continue
        chosen = follow(order, candidates, anchor, hint, scale, args.max_misses)
        for index, (pose, score) in chosen.items():
            px[index], conf[index] = pose, score
            valid[index] = score >= VALID_CONF
    for flank in (pre, post):
        interpolated[flank] = interpolate_gaps(px, conf, valid, flank, args.max_gap)[
            flank
        ]
    flank = ~from_event
    # Frames without both shoulders cannot be normalised; drop them, as upstream.
    shoulders = valid[:, 5] & valid[:, 6]
    valid[flank & ~shoulders] = False
    px[~valid] = np.nan
    root = np.full((length, 2), np.nan, np.float32)
    root[span] = event["root_px"][: stop - start]
    root[flank & shoulders] = ((px[:, 5] + px[:, 6]) / 2)[flank & shoulders]
    norm = (px - root[:, None]) / scale
    norm[span] = event["keypoints_norm"][: stop - start]
    norm[~valid] = np.nan
    return {
        "keypoints_px": px,
        "keypoints_norm": norm.astype(np.float32),
        "kp_conf": conf,
        "valid": valid,
        "interpolated": interpolated,
        "from_event": from_event,
        "root_px": root,
        "scale_px": np.float32(scale),
        "scale_source": event["scale_source"],
        "fps": np.float32(FPS),
        "fps_in": np.float32(fps_in),
        "frame_time_s": (np.arange(length) / FPS).astype(np.float32),
        "joint_names": event["joint_names"],
        "width": event["width"],
        "height": event["height"],
        "n_frames_in": old["n_frames_in"],
        "layout": event["layout"],
        "pose_model": np.array(
            f"event: {event['pose_model']} | flanks: {CHECKPOINT[1]}"
        ),
        "old_track_trusted": np.bool_(trusted),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--event-poses", default="vitpose_133")
    parser.add_argument("--old-poses", default="ViTPoseFilled")
    parser.add_argument("--output", default="vitpose_133_ctx")
    parser.add_argument("--checkpoint", type=Path, help="local vitpose-h-wholebody.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-misses", type=int, default=12, help="frames at 25 fps")
    parser.add_argument("--max-gap", type=int, default=12, help="frames to interpolate")
    parser.add_argument("--clips", nargs="*", help="only these context clip names")
    args = parser.parse_args(argv)

    old_root = args.dataset / args.old_poses
    out_root = args.dataset / args.output
    clips = sorted(old_root.glob("*/*_ctx.npz"))
    if args.clips:
        clips = [c for c in clips if c.stem in set(args.clips)]
    search = WholeBodySearch(args.device, args.checkpoint)
    for qc in old_root.glob("*/clean_qc.csv"):
        (out_root / qc.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(qc, out_root / qc.parent.name / qc.name)
    for number, old_npz in enumerate(clips, 1):
        split, name = old_npz.parent.name, old_npz.stem
        out = out_root / split / f"{name}.npz"
        event_npz = args.dataset / args.event_poses / split / f"{name[:-4]}.npz"
        if out.exists() or not event_npz.exists():
            continue  # resumable; clips without event poses are skipped
        video = args.dataset / "full_video" / split / f"{name}.mp4"
        data = build_clip(event_npz, old_npz, video, search, args)
        np.savez_compressed(out, **data)
        flank = ~data["from_event"]
        body = data["valid"][flank, :BODY].any(1)
        print(
            f"[{number}/{len(clips)}] {name}: flank frames with pose"
            f" {int(body.sum())}/{int(flank.sum())}"
            f"{'' if data['old_track_trusted'] else ' (old track not used)'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
