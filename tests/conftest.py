import csv

import numpy as np
import pytest
import torch

from asd_contextpose import ModelConfig


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(7)


@pytest.fixture
def config():
    return ModelConfig(text_dim=8, num_joints=3, joint_features=3)


@pytest.fixture
def make_manifest(tmp_path, config):
    """Synthetic tensors are test fixtures only, never research training data."""

    def make(
        split="train",
        participant=None,
        recording=None,
        overrides=None,
        metadata=None,
        count=2,
    ):
        directory = tmp_path / str(len(list(tmp_path.iterdir())))
        directory.mkdir()
        rng = np.random.default_rng(7)
        rows = []
        for index in range(count):
            shape = (config.steps, config.num_joints, config.joint_features)
            arrays = {
                "text_embedding": rng.normal(size=config.text_dim).astype(np.float32),
                **{
                    key: rng.normal(size=shape).astype(np.float32)
                    for key in ("pre_pose", "event_pose", "post_pose")
                },
                "target": rng.normal(size=2).astype(np.float32),
            }
            arrays.update(overrides or {})
            path = directory / f"{index}.npz"
            np.savez(path, **arrays)
            rows.append(
                {
                    "path": path.name,
                    "participant_id": participant or f"p-{split}",
                    "recording_id": recording or f"r-{split}",
                    "split": split,
                    **(metadata or {}),
                }
            )
        manifest = directory / "manifest.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    return make
