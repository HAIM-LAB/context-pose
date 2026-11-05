"""Frozen facial-expression features at the model's pose steps.

For every clip and region (pre/event/post), the child's face is cropped at
each of the `steps` resampling points `prepare` uses, centred on the tracked
face keypoints, and embedded with a frozen FER ViT. Quality measures are kept
alongside so `prepare --face` can mask steps where no frontal, visible face
was seen. Pose keypoints alone are a weak frontal check (they report eyes on
lowered or hair-covered heads), so a face detector (OpenCV YuNet) also scores
each step on a wider crop. Writes one cache NPZ per clip:

    embedding [3, steps, 768]   ViT [CLS] (float16)
    probs     [3, steps, 7]     expression probabilities (FER_CLASSES)
    found     [3, steps]        face keypoints tracked near the step
    frontal   [3, steps]        nose and both eyes tracked at the step
    brightness, sharpness [3, steps]  crop grey mean, Laplacian variance
    detector  [3, steps]        YuNet score of a face at the tracked position
"""

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .prepare import FLANK_S, MIN_FACE_DETECTOR, load_annotations

FER_MODEL = "trpakov/vit-face-expression"
FER_CLASSES = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
FACE_SIDE = 0.9  # crop side in shoulder widths (checked visually)
CROP = 224
NOSE, LEFT_EYE, RIGHT_EYE = 0, 1, 2
FACE_JOINTS = 5  # COCO nose, eyes, ears
DETECTOR_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
DETECTOR_PATH = Path.home() / ".cache/asd_contextpose/face_detection_yunet_2023mar.onnx"
DETECT_SIDE, DETECT_SIZE = 2.0, 320  # detector crop: shoulder widths, pixels


def step_frames(n_frames: int, fps: float, steps: int) -> np.ndarray:
    """[3, steps] pose-frame index of each resampling step, as in `prepare`."""
    flank = round(FLANK_S * fps)
    bounds = ((0, flank), (flank, n_frames - flank), (n_frames - flank, n_frames))
    return np.stack(
        [
            start + np.round(np.linspace(0, end - start - 1, steps)).astype(int)
            for start, end in bounds
        ]
    )


def face_centre(px: np.ndarray, valid: np.ndarray, frame: int) -> np.ndarray | None:
    """Mean pixel position of the tracked face keypoints near `frame`."""
    for offset in (0, 1, -1, 2, -2):
        index = frame + offset
        if 0 <= index < len(valid):
            ok = valid[index, :FACE_JOINTS]
            if ok.sum() >= 2 and ok[:3].any():
                return px[index, :FACE_JOINTS][ok].mean(0)
    return None


def square_crop(
    frame: np.ndarray, centre: np.ndarray, side: float, pixels: int = CROP
) -> np.ndarray:
    """RGB `pixels`-wide square around `centre`, zero-padded outside the frame."""
    size = max(int(round(side)), 1)
    x0, y0 = (np.round(centre - side / 2)).astype(int)
    height, width = frame.shape[:2]
    out = np.zeros((size, size, 3), np.uint8)
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + size, width), min(y0 + size, height)
    if xb > xa and yb > ya:
        out[ya - y0 : yb - y0, xa - x0 : xb - x0] = frame[ya:yb, xa:xb]
    out = cv2.resize(out, (pixels, pixels), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def crop_quality(crop: np.ndarray) -> tuple[float, float]:
    grey = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(grey.mean()), float(cv2.Laplacian(grey, cv2.CV_64F).var())


def face_detector():
    """OpenCV YuNet, downloaded once from the OpenCV model zoo."""
    if not DETECTOR_PATH.exists():
        DETECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DETECTOR_URL, DETECTOR_PATH)
    return cv2.FaceDetectorYN.create(
        str(DETECTOR_PATH), "", (DETECT_SIZE, DETECT_SIZE), 0.1
    )


def detector_score(detector, frame: np.ndarray, centre, scale: float) -> float:
    """Best score of a detected face whose box contains the tracked face centre."""
    crop = square_crop(frame, centre, DETECT_SIDE * scale, DETECT_SIZE)
    _, faces = detector.detect(np.ascontiguousarray(crop[..., ::-1]))  # BGR
    middle = DETECT_SIZE / 2
    best = 0.0
    for x, y, w, h, *_, score in [] if faces is None else faces:
        if x <= middle <= x + w and y <= middle <= y + h:
            best = max(best, float(score))
    return best


def read_frames(video: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Decode sequentially (seeking is unreliable) and keep the wanted frames."""
    capture, frames, index = cv2.VideoCapture(str(video)), {}, 0
    while wanted and index <= max(wanted):
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = frame
        index += 1
    capture.release()
    return frames


class FaceEncoder:
    def __init__(self, device: str):
        import torch
        from transformers import AutoModelForImageClassification

        self.torch, self.device = torch, torch.device(device)
        self.model = AutoModelForImageClassification.from_pretrained(FER_MODEL)
        self.model = self.model.eval().to(self.device)

    def __call__(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float()
        x = (x / 255.0 - 0.5) / 0.5  # the model's ViTImageProcessor normalisation
        embeddings, probs = [], []
        with torch.no_grad():
            for batch in x.split(64):
                cls = self.model.vit(pixel_values=batch.to(self.device))
                cls = cls.last_hidden_state[:, 0]
                embeddings.append(cls.cpu())
                probs.append(self.model.classifier(cls).softmax(-1).cpu())
        return torch.cat(embeddings).numpy(), torch.cat(probs).numpy()


def extract_clip(pose_npz: Path, video: Path, encoder, steps: int, detector) -> dict:
    with np.load(pose_npz, allow_pickle=False) as data:
        px, valid = data["keypoints_px"], data["valid"]
        times, scale = data["frame_time_s"], float(data["scale_px"])
        fps, fps_in = float(data["fps"]), float(data["fps_in"])
        last = int(data["n_frames_in"]) - 1
    frames = step_frames(len(px), fps, steps)
    centres = {}
    for region, step in np.ndindex(frames.shape):
        centre = face_centre(px, valid, frames[region, step])
        if centre is not None:
            video_frame = min(int(round(times[frames[region, step]] * fps_in)), last)
            centres[region, step] = (video_frame, centre)
    images = read_frames(video, {frame for frame, _ in centres.values()})
    out = {
        "embedding": np.zeros((3, steps, 768), np.float16),
        "probs": np.zeros((3, steps, len(FER_CLASSES)), np.float32),
        "found": np.zeros((3, steps), bool),
        "frontal": valid[frames][..., [NOSE, LEFT_EYE, RIGHT_EYE]].all(-1),
        "brightness": np.zeros((3, steps), np.float32),
        "sharpness": np.zeros((3, steps), np.float32),
        "detector": np.zeros((3, steps), np.float32),
    }
    keys, crops = [], []
    for key, (video_frame, centre) in centres.items():
        if video_frame in images:
            crop = square_crop(images[video_frame], centre, FACE_SIDE * scale)
            keys.append(key)
            crops.append(crop)
            out["brightness"][key], out["sharpness"][key] = crop_quality(crop)
            out["detector"][key] = detector_score(
                detector, images[video_frame], centre, scale
            )
    if crops:
        embedding, probs = encoder(crops)
        for index, key in enumerate(keys):
            out["found"][key] = True
            out["embedding"][key] = embedding[index]
            out["probs"][key] = probs[index]
    out["frontal"] &= out["found"]
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--poses", default="ViTPoseFilled")
    parser.add_argument("--output", type=Path, default=Path("dataset/FaceFeatures"))
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    encoder, detector = FaceEncoder(args.device), face_detector()
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
        features = extract_clip(pose, video, encoder, args.steps, detector)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, **features)
        found = features["found"].mean()
        detected = (features["detector"] >= MIN_FACE_DETECTOR).mean()
        print(
            f"{index}/{len(annotations)} {clip}: face {found:.2f}, detected {detected:.2f}"
        )


if __name__ == "__main__":
    main()
