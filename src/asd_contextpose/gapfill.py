"""Fill frames where ViTPoseClean lost the child, by following the child's track.

Starting from the nearest frame the upstream cleaner kept, each missing frame
is searched with RT-DETR person detection and ViTPose. The person whose joints
are closest to the previous pose (in shoulder widths) is accepted; if nobody
matches, rotated copies of the frame are tried. Following stops after
`--max-misses` consecutive unmatched frames, so the track cannot drift to
another person. Output mirrors ViTPoseClean, adding a `filled` mask.

`--reacquire` is a second pass over that output for gaps too long to follow
into. It learns the colour of the child's clothing (just below the shoulders)
from frames ViTPoseClean kept, and of everyone else in those frames, then
probes each gap for a person matching the child but nobody else. Two agreeing
probes 0.2 s apart seed the track, which is then followed as above. Clips where
another person dresses too similarly are skipped.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

DETECTOR = "PekingU/rtdetr_r50vd_coco_o365"
POSE_MODEL = "usyd-community/vitpose-base-simple"
VALID_CONF = 0.3  # ViTPoseClean's validity threshold
UPPER_BODY = 11  # COCO joints 0-10: face, shoulders, elbows, wrists
MAX_LOOK_DISTANCE = 0.7  # Bhattacharyya distance between clothing histograms
ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def unrotate(points: np.ndarray, angle: int, width: int, height: int) -> np.ndarray:
    """Map [N, 2] points from a rotated frame back to the original frame."""
    x, y = points[:, 0], points[:, 1]
    if angle == 90:  # clockwise: original (x, y) -> (height - 1 - y, x)
        return np.stack([y, height - 1 - x], 1)
    if angle == 180:
        return np.stack([width - 1 - x, height - 1 - y], 1)
    if angle == 270:
        return np.stack([width - 1 - y, x], 1)
    return points


def match_distance(pose: np.ndarray, conf: np.ndarray, ref, scale: float) -> float:
    """Mean joint distance in shoulder widths over joints confident in both."""
    ref_pose, ref_ok = ref
    shared = (conf >= VALID_CONF) & ref_ok
    if shared.sum() < 3:
        return np.inf
    return float(np.linalg.norm(pose[shared] - ref_pose[shared], axis=1).mean() / scale)


def torso_hist(bgr: np.ndarray, pose: np.ndarray, ok: np.ndarray) -> np.ndarray | None:
    """Hue/saturation histogram of the torso: a shoulder-wide box below the shoulders."""
    if not (ok[5] and ok[6]):
        return None
    left, right = pose[5], pose[6]
    width = float(np.linalg.norm(right - left))
    if width < 4:
        return None
    mid = (left + right) / 2
    down = np.array([-(right - left)[1], (right - left)[0]]) / width
    face = pose[:5][ok[:5]]
    toward_face = np.dot(face.mean(0) - mid, down) if len(face) else -down[1]
    if toward_face > 0:
        down = -down
    corners = np.array(
        [left, right, right + down * width * 1.2, left + down * width * 1.2]
    )
    mask = np.zeros(bgr.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 1)
    if mask.sum() < 50:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, None).flatten()


def look_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


class PoseSearch:
    def __init__(self, device: str):
        from transformers import (
            AutoImageProcessor,
            AutoProcessor,
            RTDetrForObjectDetection,
            VitPoseForPoseEstimation,
        )

        self.device = device
        self.det_proc = AutoProcessor.from_pretrained(DETECTOR)
        self.det = RTDetrForObjectDetection.from_pretrained(DETECTOR).to(device).eval()
        self.pose_proc = AutoImageProcessor.from_pretrained(POSE_MODEL)
        self.pose = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL)
        self.pose = self.pose.to(device).eval()

    @torch.no_grad()
    def candidates(self, bgr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """All people in the frame as ([11, 2] pixels, [11] confidence)."""
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        size = torch.tensor([(image.height, image.width)])
        inputs = self.det_proc(images=image, return_tensors="pt").to(self.device)
        found = self.det_proc.post_process_object_detection(
            self.det(**inputs), target_sizes=size, threshold=VALID_CONF
        )[0]
        boxes = found["boxes"][found["labels"] == 0].cpu().numpy()
        if not len(boxes):
            return []
        boxes[:, 2:] -= boxes[:, :2]  # xyxy -> xywh
        inputs = self.pose_proc(image, boxes=[boxes], return_tensors="pt")
        people = self.pose_proc.post_process_pose_estimation(
            self.pose(**inputs.to(self.device)), boxes=[boxes]
        )[0]
        return [
            (
                p["keypoints"][:UPPER_BODY].cpu().numpy(),
                p["scores"][:UPPER_BODY].cpu().numpy(),
            )
            for p in people
        ]

    def find(self, bgr, ref, scale, tolerance):
        """Best match to `ref` within tolerance, trying rotations if needed."""
        height, width = bgr.shape[:2]
        for angle in (0, *ROTATIONS):
            frame = bgr if angle == 0 else cv2.rotate(bgr, ROTATIONS[angle])
            scored = []
            for pose, conf in self.candidates(frame):
                pose = unrotate(pose, angle, width, height)
                scored.append((match_distance(pose, conf, ref, scale), pose, conf))
            if scored:
                best = min(scored, key=lambda s: s[0])
                if best[0] <= tolerance:
                    return best[1], best[2]
        return None

    def find_by_look(self, bgr, look, threshold, scale):
        """The one person dressed like `look`, trying rotations if needed."""
        height, width = bgr.shape[:2]
        for angle in (0, *ROTATIONS):
            frame = bgr if angle == 0 else cv2.rotate(bgr, ROTATIONS[angle])
            scored = []
            for pose, conf in self.candidates(frame):
                ok = conf >= VALID_CONF
                shoulder = np.linalg.norm(pose[5] - pose[6])
                if ok.sum() < 5 or not 0.4 * scale <= shoulder <= 2.5 * scale:
                    continue
                hist = torso_hist(frame, pose, ok)
                if hist is not None:
                    pose = unrotate(pose, angle, width, height)
                    scored.append((look_distance(hist, look), pose, conf))
            scored.sort(key=lambda s: s[0])
            if scored and scored[0][0] <= threshold:
                if len(scored) > 1 and scored[1][0] <= threshold:
                    return None  # two people dressed alike
                return scored[0][1], scored[0][2]
        return None


def learn_look(read, search, px, valid, scale, samples=20):
    """The child's clothing histogram and a threshold separating it from others.

    Returns (look, threshold), or (None, nan) if the child cannot be told apart.
    """
    usable = np.flatnonzero(valid[:, 5] & valid[:, 6] & (valid.sum(1) >= 5))
    if len(usable) < 5:
        return None, np.nan
    child, others = [], []
    for n, index in enumerate(
        usable[np.linspace(0, len(usable) - 1, samples).astype(int)]
    ):
        bgr = read(index)
        if bgr is None:
            continue
        hist = torso_hist(bgr, px[index], valid[index])
        if hist is not None:
            child.append(hist)
        if n % 2 == 0:  # other people, told apart from the child by pose
            for pose, conf in search.candidates(bgr):
                ok = conf >= VALID_CONF
                if match_distance(pose, conf, (px[index], valid[index]), scale) > 1.0:
                    other = torso_hist(bgr, pose, ok)
                    if other is not None:
                        others.append(other)
    if len(child) < 5:
        return None, np.nan
    look = cv2.normalize(np.mean(child, 0), None).flatten()
    spread = np.percentile([look_distance(h, look) for h in child], 90)
    nearest_other = min((look_distance(h, look) for h in others), default=np.inf)
    if nearest_other <= spread:
        return None, np.nan
    threshold = spread + 0.05 if not others else (spread + nearest_other) / 2
    # Above this, clothing is too variable to identify the child: on review,
    # both clips re-acquired at 0.81-0.82 had locked onto a different child.
    if threshold > MAX_LOOK_DISTANCE:
        return None, np.nan
    return look, threshold


def fill_clip(
    npz: Path, video: Path, search: PoseSearch, max_misses: int, reacquire: bool = False
) -> dict:
    with np.load(npz, allow_pickle=False) as source:
        data = dict(source)
    px, conf = data["keypoints_px"], data["kp_conf"]
    filled = data["filled"].copy() if "filled" in data else np.zeros_like(data["valid"])
    valid = data["valid"] & ~filled  # upstream observations only
    first_pass = "filled" not in data
    before = filled.any(1)
    scale, fps_in = float(data["scale_px"]), float(data["fps_in"])
    seen = valid.any(1)
    cap = cv2.VideoCapture(str(video))

    def read(index: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(index / 25 * fps_in))
        ok, bgr = cap.read()
        return bgr if ok else None

    def follow(order):
        """Walk `order` (frame indices) from a seen anchor into the gap."""
        ref, misses = None, 0
        for index in order:
            if seen[index] or filled[index].any():
                ref, misses = (px[index].copy(), valid[index] | filled[index]), 0
                continue
            if ref is None or misses > max_misses:
                continue
            bgr = read(index)
            # Allow more movement the longer the child has been unmatched.
            tolerance = min(0.5 + 0.25 * misses, 2.0)
            hit = None if bgr is None else search.find(bgr, ref, scale, tolerance)
            if hit is None:
                misses += 1
                continue
            pose, score = hit
            ok = score >= VALID_CONF
            px[index], conf[index], filled[index] = pose, score, ok
            ref = (np.where(ok[:, None], pose, ref[0]), ref[1] | ok)
            misses = 0

    if first_pass:
        follow(range(len(seen)))
        follow(range(len(seen) - 1, -1, -1))
    seeded = np.zeros(len(seen), bool)
    if reacquire:
        look, threshold = learn_look(read, search, px, valid, scale)
        data["reacquire_threshold"] = np.float32(threshold)
        previous = None
        for index in range(0, len(seen), 5):  # probe gaps every 0.2 s
            if seen[index] or filled[index].any() or look is None:
                previous = None
                continue
            bgr = read(index)
            hit = (
                None
                if bgr is None
                else search.find_by_look(bgr, look, threshold, scale)
            )
            if hit is not None and previous is not None:
                ref = (previous[1], previous[2] >= VALID_CONF)
                if match_distance(hit[0], hit[1], ref, scale) <= 1.0:
                    for i, pose, score in (previous, (index, *hit)):
                        px[i], conf[i], filled[i] = pose, score, score >= VALID_CONF
                        seeded[i] = True
            previous = None if hit is None else (index, *hit)
        if seeded.any():
            follow(range(len(seen)))
            follow(range(len(seen) - 1, -1, -1))
    cap.release()

    # Normalise like ViTPoseClean: shoulder-midpoint origin, clip shoulder scale.
    shoulders = (valid | filled)[:, 5] & (valid | filled)[:, 6]
    newly = filled.any(1) & shoulders
    filled &= newly[:, None]  # frames without both shoulders cannot be normalised
    root = data["root_px"].copy()
    root[newly] = (px[newly, 5] + px[newly, 6]) / 2
    norm = data["keypoints_norm"].copy()
    norm[filled] = ((px - root[:, None]) / scale)[filled]
    data.update(
        keypoints_px=px.astype(np.float32),
        keypoints_norm=norm.astype(np.float32),
        kp_conf=conf.astype(np.float32),
        valid=valid | filled,
        root_px=root.astype(np.float32),
        filled=filled,
        fill_models=np.array(f"{DETECTOR}|{POSE_MODEL}"),
    )
    if reacquire:
        data.update(reacquired=filled.any(1) & ~before, seeded=seeded & filled.any(1))
    return data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", default="ViTPoseFilled")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-misses", type=int, default=12, help="frames at 25 fps")
    parser.add_argument("--clips", nargs="*", help="only these clip names")
    parser.add_argument(
        "--reacquire",
        action="store_true",
        help="second pass over --output: re-find the child by clothing in remaining gaps",
    )
    args = parser.parse_args(argv)

    source = args.dataset / (args.output if args.reacquire else "ViTPoseClean")
    search = PoseSearch(args.device)
    if not args.reacquire:
        for qc in source.glob("*/clean_qc.csv"):
            target = args.dataset / args.output / qc.parent.name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(qc, target / qc.name)
    clips = sorted(source.glob("*/*.npz"))
    if args.clips:
        clips = [c for c in clips if c.stem in set(args.clips)]
    for number, npz in enumerate(clips, 1):
        out = args.dataset / args.output / npz.parent.name / npz.name
        with np.load(npz) as clip:
            before = clip["valid"].any(1)
            done = "reacquired" in clip if args.reacquire else out.exists()
        if done or (args.reacquire and before.all()):
            continue  # resumable
        video = args.dataset / "full_video" / npz.parent.name / f"{npz.stem}.mp4"
        data = fill_clip(npz, video, search, args.max_misses, args.reacquire)
        np.savez_compressed(out, **data)
        gained = int((data["valid"].any(1) & ~before).sum())
        note = ""
        if args.reacquire:
            threshold = float(data["reacquire_threshold"])
            note = " (child not distinguishable)" if np.isnan(threshold) else ""
            note += f" seeds {int(data['seeded'].sum())}"
        print(
            f"[{number}/{len(clips)}] {npz.stem}: missing {int((~before).sum())}"
            f" -> {int((~data['valid'].any(1)).sum())} frames (+{gained}){note}",
            flush=True,
        )


if __name__ == "__main__":
    main()
