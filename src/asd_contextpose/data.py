"""Validated NPZ samples and participant-independent manifest boundaries."""

import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .model import FACE_KEYS, ModelConfig

POSE_KEYS = ("pre_pose", "event_pose", "post_pose")


class PoseDataset(Dataset):
    """Paths are relative to the manifest; target stays separate from inputs."""

    def __init__(self, manifest: str | Path, config: ModelConfig, split: str):
        self.manifest = Path(manifest).resolve()
        self.config = config
        self.split = split
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        with self.manifest.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"path", "participant_id", "recording_id", "split"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"manifest requires {sorted(required)}")
            rows = list(reader)
        self.rows = []
        self.excluded_review = 0
        seen = set()
        for row in rows:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    "manifest row has a different number of fields than the header"
                )
            if any(not row[key].strip() for key in required):
                raise ValueError("manifest contains empty required fields")
            row = {
                key: value.strip() if isinstance(value, str) else value
                for key, value in row.items()
            }
            if row["split"] != split:
                raise ValueError(f"expected split {split}, got {row['split']}")
            review = (row.get("needs_review") or "false").lower()
            if review not in {"true", "false", "1", "0"}:
                raise ValueError("needs_review must be true/false or 1/0")
            if review in {"true", "1"}:
                self.excluded_review += 1
                continue
            path = (self.manifest.parent / row["path"]).resolve()
            if path in seen:
                raise ValueError(f"duplicate sample path: {path}")
            seen.add(path)
            row["path"] = str(path)
            self._validate_timing(row)
            self.rows.append(row)
        if not self.rows:
            raise ValueError("manifest has no eligible samples")
        orders = {row.get("joint_order", "") for row in self.rows}
        if len(orders) != 1:
            raise ValueError("joint_order must be consistent for every eligible sample")
        self.joint_order = orders.pop()
        if self.joint_order:
            joints = self.joint_order.split("|")
            if len(joints) != config.num_joints or len(set(joints)) != len(joints):
                raise ValueError(
                    "joint_order must contain J distinct names separated by |"
                )
        # Fail before training, even if a malformed sample is in the final batch.
        for index in range(len(self)):
            self[index]

    @staticmethod
    def _validate_timing(row: dict) -> None:
        intervals = {}
        for region in ("event", "post", "outcome"):
            start, end = row.get(f"{region}_start"), row.get(f"{region}_end")
            if not start and not end:
                continue
            if not start or not end:
                raise ValueError(
                    f"{region}_start and {region}_end must be supplied together"
                )
            start, end = float(start), float(end)
            if not math.isfinite(start) or not math.isfinite(end) or start >= end:
                raise ValueError(f"invalid {region} interval")
            intervals[region] = (start, end)
        if "outcome" in intervals and "post" not in intervals:
            raise ValueError("outcome timing requires post timing for leakage checks")
        if "event" in intervals and "post" in intervals:
            if intervals["post"][0] < intervals["event"][1]:
                raise ValueError("post input must start at or after the event end")
        if row.get("event_duration"):
            duration = float(row["event_duration"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("event_duration must be finite and positive")
            if "event" in intervals and not math.isclose(
                duration,
                intervals["event"][1] - intervals["event"][0],
                rel_tol=1e-5,
                abs_tol=1e-6,
            ):
                raise ValueError("event_duration disagrees with event endpoints")
        if "outcome" in intervals:
            post, outcome = intervals["post"], intervals["outcome"]
            if max(post[0], outcome[0]) < min(post[1], outcome[1]):
                raise ValueError(
                    "post-event input overlaps outcome; truncate/exclude upstream"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        c = self.config
        pose_shape = (c.steps, c.num_joints, c.joint_features)
        # The conditioning vector is CLIP text or annotations, per model variant.
        input_keys = (c.condition_key, *POSE_KEYS, *(FACE_KEYS if c.face_dim else ()))
        shapes = {
            c.condition_key: (c.text_dim,),
            **dict.fromkeys(POSE_KEYS, pose_shape),
        }
        if c.face_dim:  # face features plus the quality-gate mask column
            shapes.update(dict.fromkeys(FACE_KEYS, (c.steps, c.face_dim + 1)))
        shapes["target"] = (2,)
        if c.objective == "votes":
            shapes["votes"] = (2,)
        row = self.rows[index]
        with np.load(row["path"], allow_pickle=False) as sample:
            tensors = {}
            for key, shape in shapes.items():
                if key not in sample:
                    raise ValueError(f"{row['path']}: missing {key}")
                array = sample[key]
                if array.shape != shape or array.dtype != np.float32:
                    raise ValueError(f"{row['path']}: {key} must be float32 {shape}")
                if not np.isfinite(array).all():
                    raise ValueError(f"{row['path']}: {key} contains non-finite values")
                tensors[key] = torch.from_numpy(array.copy())
        return {
            "inputs": {key: tensors[key] for key in input_keys},
            "target": tensors["target"],
            **({"votes": tensors["votes"]} if "votes" in tensors else {}),
            "participant_id": row["participant_id"],
            "recording_id": row["recording_id"],
        }

    def groups(self) -> dict[str, list[str]]:
        return {
            key: sorted({row[key] for row in self.rows})
            for key in ("participant_id", "recording_id", "path")
        }


def assert_group_separation(*datasets: PoseDataset) -> None:
    for index, left in enumerate(datasets):
        for right in datasets[index + 1 :]:
            if left.config != right.config or left.joint_order != right.joint_order:
                raise ValueError(
                    "datasets must have matching dimensions and joint_order"
                )
            assert_disjoint(left.groups(), right.groups())


def assert_disjoint(left: dict, right: dict) -> None:
    for key in ("participant_id", "recording_id", "path"):
        overlap = set(left[key]) & set(right[key])
        if overlap:
            raise ValueError(f"split leakage: overlapping {key}: {sorted(overlap)}")
