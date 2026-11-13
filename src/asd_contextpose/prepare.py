"""Convert ViTPose context clips and consensus annotations into model NPZs.

Each clip is [5 s pre-event] + [event] + [5 s post-event] on the clip clock.
Poses use the dataset's shoulder-centred, shoulder-width-scaled coordinates
(`keypoints_norm`) with features [x, y, confidence]. The conditioning vector is
the frozen, L2-normalised CLIP ViT-L/14 embedding of each clip's natural event
text, or the five consensus context indicators with `--text-encoder indicators`.
Every NPZ also holds `votes` [2]: the share of annotators voting positive
valence and high arousal (see `asd_contextpose.votes`). With `--face`, per-step
face features from `asd_contextpose.face` are added as `pre_face`/`event_face`/
`post_face` [steps, D + 1]: the features, zeroed where the quality gate fails,
then the gate's 0/1 mask.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import openpyxl

from . import votes as annotator_votes

FLANK_S = 5.0
MAX_SPEED = 20.0  # shoulder widths per second; faster steps are tracking jumps
TEXT_CSV = "ASD_ContextPose_Main_Event_Text_Natural.csv"
CLIP_MODEL = "openai/clip-vit-large-patch14"
INDICATORS = (
    "social_interaction",
    "auditory_event",
    "activity_object_change",
    "visual_media_event",
    "no_event",
)
CONTEXT = INDICATORS[:4]  # annotation-model input; no_event is all four at zero
# Header prefixes of a single annotator's sheet (cells are `value:confidence`).
SINGLE_ANNOTATOR_COLUMNS = {
    "Emotional Effect (Valence": "valence",
    "Emotional Effect (Arousal": "arousal",
    "social interaction": "social_interaction",
    "auditory event": "auditory_event",
    "activity/object change": "activity_object_change",
    "visual/media event": "visual_media_event",
    "No Event": "no_event",
}
# The same footage filed under two categories: one participant and recording.
DUPLICATE_VIDEOS = {"183_v_HandAction_05_0": "109_v_ArmFlapping_59"}
FACE_FEATURES = ("embedding", "probs")
# Face quality gate: frontal face (nose and both eyes tracked, and YuNet finds
# a face there), not too dark, not too blurred. Brightness and sharpness are on
# the 224 px crop's grey levels. Below 0.8, YuNet's crops were mostly hair or
# the top of a lowered head (checked visually).
MIN_FACE_DETECTOR = 0.8
MIN_FACE_BRIGHTNESS = 40.0
MIN_FACE_SHARPNESS = 10.0  # Laplacian variance; small faces are upsampled


def load_annotations(path: Path) -> dict[str, dict]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = workbook["Final_detailed"].iter_rows(values_only=True)
    header = next(rows)
    records = (dict(zip(header, row)) for row in rows)
    return {r["clip_name"]: r for r in records if r["clip_name"]}


def load_single_annotator(
    path: Path, annotator: str
) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """One annotator's `value:confidence` sheet, as records like `load_annotations`.

    Only the value before the colon is used. Indicators become 0/1 columns; the
    valence/arousal cells become that annotator's votes, and the label is the
    vote on the [-1, 1] scale. Clips are dropped if No event is 1 alongside
    another indicator, or if valence or arousal is unknown.
    """
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = workbook.worksheets[0].iter_rows(values_only=True)
    header = next(rows)
    columns = {}
    for index, name in enumerate(header):
        for prefix, key in SINGLE_ANNOTATOR_COLUMNS.items():
            if name and str(name).startswith(prefix):
                columns[key] = index
    missing = set(SINGLE_ANNOTATOR_COLUMNS.values()) - set(columns)
    if missing:
        raise ValueError(f"{path}: no column for {sorted(missing)}")
    records, excluded = {}, []
    for row in rows:
        if not row[0]:
            continue
        clip = row[0]
        cells = {key: row[i] for key, i in columns.items()}
        value = {key: str(cell).split(":", 1)[0].strip() for key, cell in cells.items()}
        record = {"clip_name": clip, "Category": row[1], "FolderName": row[2]}
        record.update({k: int(value[k]) for k in INDICATORS})
        if record["no_event"] and any(record[k] for k in CONTEXT):
            excluded.append((clip, "no_event is 1 alongside another indicator"))
            continue
        for target in annotator_votes.TARGETS:
            vote = annotator_votes.VOTE_VALUES[target].get(value[target])
            if vote is None:
                break
            record[target] = float(vote)
            record[f"{target}__{annotator}"] = cells[target]
        else:
            records[clip] = {**record, "needs_review": None}
            continue
        excluded.append((clip, f"unknown {target} ({value[target]})"))
    return records, excluded


def load_event_texts(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["clip_name"]: row["event_text"] for row in csv.DictReader(handle)}


def encode_clip_text(texts: list[str]) -> tuple[dict[str, np.ndarray], str]:
    """Embed each distinct text once; returns vectors and a pinned model id."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    from transformers.utils import logging

    # The checkpoint also holds the vision tower; its "unexpected keys" are noise.
    logging.set_verbosity_error()
    # The snapshot directory name is the commit hash of the files actually loaded.
    revision = Path(hf_hub_download(CLIP_MODEL, "config.json")).parent.name
    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL)
    model = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL).eval()
    unique = sorted(set(texts))
    with torch.no_grad():
        tokens = tokenizer(unique, padding=True, return_tensors="pt")
        embeds = torch.nn.functional.normalize(model(**tokens).text_embeds, dim=-1)
    vectors = dict(zip(unique, embeds.numpy().astype(np.float32)))
    return vectors, f"{CLIP_MODEL}@{revision}"


def load_qc(pose_root: Path) -> dict[str, bool]:
    qc = {}
    for path in pose_root.glob("*/clean_qc.csv"):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                qc[row["clip"]] = row["ok_head"] == "True" and row["ok_arms"] == "True"
    return qc


def resample(features: np.ndarray, valid: np.ndarray, steps: int) -> np.ndarray:
    """Linearly resample [T, J, F] frames onto `steps` points spanning the region.

    Grid points use the two bracketing frames; if only one is valid it is used
    alone, and if neither is, the point stays zero (confidence zero). Gaps the
    upstream cleaner left unfilled are therefore not bridged here.
    """
    position = np.linspace(0, len(features) - 1, steps)
    i0 = np.floor(position).astype(int)
    i1 = np.minimum(i0 + 1, len(features) - 1)
    w = (position - i0)[:, None, None]
    lerp = (1 - w) * features[i0] + w * features[i1]
    v0, v1 = valid[i0][..., None], valid[i1][..., None]
    # Invalid frames are already zero, so the last fallback yields zeros.
    out = np.where(v0 & v1, lerp, np.where(v0, features[i0], features[i1]))
    return out.astype(np.float32)


def joint_speed(norm: np.ndarray, valid: np.ndarray, fps: float) -> np.ndarray:
    """[T, J] joint speed in shoulder widths/s, from the frame before (or after).

    Steps faster than MAX_SPEED are tracking jumps and are clipped; joints with
    no valid neighbouring frame get zero.
    """
    step = np.linalg.norm(np.diff(norm, axis=0), axis=-1) * fps
    step = np.where(valid[1:] & valid[:-1], np.minimum(step, MAX_SPEED), np.nan)
    before = np.vstack([np.full((1, step.shape[1]), np.nan), step])
    after = np.vstack([step, np.full((1, step.shape[1]), np.nan)])
    return np.nan_to_num(np.where(np.isnan(before), after, before))


def bin_mean(values: np.ndarray, valid: np.ndarray, width: int) -> np.ndarray:
    """Centred moving mean over `width` frames of the valid entries of [T, J]."""

    def smooth(x: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(np.convolve, 0, x, np.ones(width), "same")

    count = smooth(valid.astype(float))
    total = smooth(np.where(valid, values, 0.0))
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


def convert_clip(npz: Path, steps: int, speed: bool = False) -> dict[str, np.ndarray]:
    """Regions as [steps, J, F] with F = [x, y, confidence] (+ speed if asked).

    Speed is measured on the 25 fps frames and averaged over each resampling
    bin, so it stays in shoulder widths per second whatever the region length.
    """
    with np.load(npz, allow_pickle=False) as d:
        norm, conf, valid = d["keypoints_norm"], d["kp_conf"], d["valid"]
        fps, joints = float(d["fps"]), d["joint_names"]
        filled = d["filled"] if "filled" in d else np.zeros_like(valid)
    flank = round(FLANK_S * fps)
    if len(norm) <= 2 * flank:
        raise ValueError(f"clip is {len(norm) / fps:.2f} s, leaving no event frames")
    valid = valid & np.isfinite(norm).all(-1)
    features = np.concatenate([norm, np.clip(conf, 0, 1)[..., None]], axis=-1)
    features = np.where(valid[..., None], features, 0)
    speeds = joint_speed(norm, valid, fps) if speed else None
    regions = {
        "pre_pose": slice(0, flank),
        "event_pose": slice(flank, len(norm) - flank),
        "post_pose": slice(len(norm) - flank, len(norm)),
    }
    out = {}
    for key, frames in regions.items():
        if not valid[frames].any():
            raise ValueError(f"{key} has no observed keypoints")
        region = features[frames]
        if speed:
            width = max(1, round(len(region) / steps))
            mean_speed = bin_mean(speeds[frames], valid[frames], width)
            region = np.concatenate([region, mean_speed[..., None]], axis=-1)
            region = np.where(valid[frames][..., None], region, 0)
        out[key] = resample(region, valid[frames], steps)
    duration = len(norm) / fps
    out["timing"] = {
        "event_start": FLANK_S,
        "event_end": duration - FLANK_S,
        "event_duration": duration - 2 * FLANK_S,
        "post_start": duration - FLANK_S,
        "post_end": duration,
    }
    out["joint_order"] = "|".join(joints.tolist())
    out["filled_frac"] = float(filled.any(1).mean())
    return out


def face_regions(cache: Path | None, features: str, steps: int) -> dict:
    """Gated per-step face features; an absent cache means no usable face."""
    if cache is None or not cache.exists():
        width = 768 if features == "embedding" else 7
        empty = np.zeros((steps, width + 1), np.float32)
        return {key: empty for key in ("pre_face", "event_face", "post_face")}
    with np.load(cache, allow_pickle=False) as data:
        values = data[features].astype(np.float32)
        if values.shape[1] != steps:
            raise ValueError(f"{cache}: face cache has {values.shape[1]} steps")
        keep = (
            data["frontal"]
            & (data["detector"] >= MIN_FACE_DETECTOR)
            & (data["brightness"] >= MIN_FACE_BRIGHTNESS)
            & (data["sharpness"] >= MIN_FACE_SHARPNESS)
        )
    gated = np.concatenate(
        [np.where(keep[..., None], values, 0), keep[..., None]], -1
    ).astype(np.float32)
    return dict(zip(("pre_face", "event_face", "post_face"), gated))


def video_id(clip: str) -> str:
    """The clip's video, with known duplicates mapped to one recording."""
    video = clip.rsplit("_seg", 1)[0]
    return DUPLICATE_VIDEOS.get(video, video)


def assign_val(clips: list[dict], fraction: float, seed: int) -> set[str]:
    """Hold out whole videos from train, stratified by category."""
    rng = np.random.default_rng(seed)
    by_category: dict[str, set[str]] = {}
    for clip in clips:
        by_category.setdefault(clip["Category"], set()).add(video_id(clip["clip_name"]))
    held = set()
    for category in sorted(by_category):
        videos = sorted(by_category[category])
        count = max(1, round(fraction * len(videos))) if len(videos) > 1 else 0
        held.update(rng.permutation(videos)[:count].tolist())
    return held


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("data/vitpose"))
    parser.add_argument(
        "--poses",
        default="ViTPoseFilled",
        help="pose folder under --dataset (ViTPoseClean for the unfilled poses)",
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--speed",
        action="store_true",
        help="add a 4th pose feature: joint speed (shoulder widths/s at 25 fps)",
    )
    parser.add_argument(
        "--face",
        type=Path,
        metavar="DIR",
        help="face cache from `python -m asd_contextpose.face`; adds face inputs",
    )
    parser.add_argument("--face-features", choices=FACE_FEATURES, default="embedding")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-qc",
        action="store_true",
        help="drop clips that failed the pose cleaner's ok_head/ok_arms checks",
    )
    parser.add_argument(
        "--text-encoder",
        choices=("clip-vit-l-14", "indicators"),
        default="clip-vit-l-14",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        metavar="XLSX",
        help="one annotator's value:confidence sheet (e.g. ContextAnnotation_Olivia_v3"
        ".xlsx under --dataset) instead of the three-annotator consensus",
    )
    parser.add_argument(
        "--annotator", default="Olivia", help="name for --annotations' votes"
    )
    args = parser.parse_args(argv)
    if not 0 <= args.val_fraction < 1:
        parser.error("val-fraction must be in [0, 1)")
    if args.annotations and args.text_encoder != "indicators":
        parser.error(
            "the event texts follow the consensus; use --text-encoder indicators"
        )

    pose_root = args.dataset / args.poses
    excluded = []
    if args.annotations:
        annotations, excluded = load_single_annotator(
            args.dataset / args.annotations, args.annotator
        )
    else:
        annotations = load_annotations(args.dataset / "Final_Annotation_v2.xlsx")
    qc = load_qc(pose_root)
    if args.text_encoder == "indicators":
        texts, encoder = {}, "indicators:" + "|".join(INDICATORS)
    else:
        texts = load_event_texts(args.dataset / TEXT_CSV)
        if set(texts) != set(annotations):
            parser.error(f"{TEXT_CSV} clips differ from the annotation workbook")
        vectors, encoder = encode_clip_text(list(texts.values()))
    val_videos = assign_val(
        [a for a in annotations.values() if a["FolderName"] == "train"],
        args.val_fraction,
        args.seed,
    )
    manifests: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for clip, ann in sorted(annotations.items()):
        source = pose_root / ann["FolderName"] / f"{clip}.npz"
        if args.require_qc and not qc[clip]:
            excluded.append((clip, "failed pose QC"))
            continue
        try:
            sample = convert_clip(source, args.steps, args.speed)
        except ValueError as error:
            excluded.append((clip, str(error)))
            continue
        split = ann["FolderName"]
        if split == "train" and video_id(clip) in val_videos:
            split = "val"
        target = args.output / split / f"{clip}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if texts:
            text_embedding = vectors[texts[clip]]
        else:
            text_embedding = np.array([ann[k] for k in INDICATORS], dtype=np.float32)
        faces = {}
        if args.face:
            cache = args.face / ann["FolderName"] / f"{clip}.npz"
            faces = face_regions(cache, args.face_features, args.steps)
        clip_votes = {
            t: annotator_votes.parse_votes(ann, t) for t in annotator_votes.TARGETS
        }
        np.savez(
            target,
            **faces,
            votes=np.array(
                [annotator_votes.soft_target(clip_votes[t], t) for t in clip_votes],
                dtype=np.float32,
            ),
            text_embedding=text_embedding,
            context=np.array([ann[k] for k in CONTEXT], dtype=np.float32),
            pre_pose=sample["pre_pose"],
            event_pose=sample["event_pose"],
            post_pose=sample["post_pose"],
            target=np.array([ann["valence"], ann["arousal"]], dtype=np.float32),
        )
        manifests[split].append(
            {
                "path": f"{split}/{clip}.npz",
                # No participant IDs are available; the video is the finest group.
                "participant_id": video_id(clip),
                "recording_id": video_id(clip),
                "split": split,
                "needs_review": str(ann["needs_review"] == "Y").lower(),
                "joint_order": sample["joint_order"],
                **{k: f"{v:.2f}" for k, v in sample["timing"].items()},
                "category": ann["Category"],
                # Lookup key: the indicators that are 1 (residual/baseline grouping).
                "context_combination": "+".join(k for k in CONTEXT if ann[k])
                or "no_event",
                "pose_qc_ok": str(qc[clip]).lower(),
                "pose_filled_frac": f"{sample['filled_frac']:.3f}",
                "text_encoder": encoder,
                **{
                    f"{t}_votes": annotator_votes.encode(v)
                    for t, v in clip_votes.items()
                },
                **(
                    {
                        "face_kept_frac": f"{np.mean([f[:, -1] for f in faces.values()]):.3f}"
                    }
                    if faces
                    else {}
                ),
                **({"event_text": texts[clip]} if texts else {}),
            }
        )
    for split, rows in manifests.items():
        with (args.output / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"{split}: {len(rows)} clips, {len({r['recording_id'] for r in rows})} videos"
        )
    print(f"text_dim={len(text_embedding)} ({encoder})")
    for clip, reason in excluded:
        print(f"excluded {clip}: {reason}")


if __name__ == "__main__":
    main()
