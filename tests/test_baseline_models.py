import json

import numpy as np
import pytest
import torch

from asd_contextpose import ModelConfig, build_model
from asd_contextpose.baseline_models import (
    AnnotationRecurrentBaseline,
    RecurrentBaseline,
)
from asd_contextpose.train import main

CELLS = ["gru", "bilstm"]


def inputs(config, batch=2):
    return [torch.randn(batch, config.text_dim)] + [
        torch.randn(batch, config.steps, config.num_joints, config.joint_features)
        for _ in range(3)
    ]


def faces(config, batch=2, mask=1.0):
    features = torch.randn(batch, config.steps, config.face_dim)
    gate = torch.full((batch, config.steps, 1), mask)
    return [torch.cat([features, gate], -1) for _ in range(3)]


@pytest.mark.parametrize("cell", CELLS)
@pytest.mark.parametrize("batch", [1, 3])
def test_forward_backward(cell, batch):
    model = build_model(ModelConfig(8, 3, 3, width=16, model=f"text_{cell}"))
    assert type(model) is RecurrentBaseline
    pred = model(*inputs(model.config, batch))
    assert pred.shape == (batch, 2) and torch.isfinite(pred).all()
    pred.square().sum().backward()
    for parameter in (model.text_projection.weight, model.step_projection[0].weight):
        assert parameter.grad.abs().sum() > 0
    assert all(p.grad is not None for p in model.rnn.parameters())


@pytest.mark.parametrize("cell", CELLS)
def test_annotation_baselines_take_four_context_features(cell):
    model = build_model(ModelConfig(4, 3, 3, width=16, model=f"annotation_{cell}"))
    assert isinstance(model, AnnotationRecurrentBaseline)
    assert model.config.condition_key == "context"
    assert model(*inputs(model.config)).shape == (2, 2)
    with pytest.raises(ValueError, match="4 context features"):
        ModelConfig(8, 3, 3, model=f"annotation_{cell}")
    with pytest.raises(ValueError, match="context must have shape"):
        model(torch.zeros(2, 5), *inputs(model.config)[1:])


@pytest.mark.parametrize("cell", CELLS)
def test_sequence_order_matters(cell):
    """Unlike a pooled model, reversing the steps changes the prediction."""
    model = build_model(ModelConfig(8, 3, 3, width=16, model=f"text_{cell}")).eval()
    text, *poses = inputs(model.config)
    flipped = [pose.flip(1) for pose in reversed(poses)]
    assert not torch.allclose(model(text, *poses), model(text, *flipped))


@pytest.mark.parametrize("cell", CELLS)
def test_ablations_ignore_the_dropped_input(cell):
    def config(**kwargs):
        return ModelConfig(
            8, 3, 3, width=16, model=f"text_{cell}", face_dim=5, **kwargs
        )

    torch.manual_seed(0)
    pose_only = build_model(config(inputs="pose")).eval()
    text, *poses = inputs(pose_only.config)
    face = faces(pose_only.config)
    base = pose_only(text, *poses, *face)
    assert torch.allclose(base, pose_only(text * 0 + 9, *poses, *face))
    for ablation in ("text", "face"):
        model = build_model(config(inputs=ablation)).eval()
        base = model(text, *poses, *face)
        assert torch.allclose(base, model(text, *(p + 1 for p in poses), *face))
    face_only = build_model(config(inputs="face")).eval()
    assert torch.allclose(
        face_only(text, *poses, *face), face_only(text + 1, *poses, *face)
    )


@pytest.mark.parametrize("cell", CELLS)
def test_masked_face_steps_are_ignored(cell):
    model = build_model(
        ModelConfig(8, 3, 3, width=16, model=f"text_{cell}", face_dim=5)
    ).eval()
    text, *poses = inputs(model.config)
    hidden = faces(model.config, mask=0.0)
    noisy = [torch.cat([f[..., :-1] + 5, f[..., -1:]], -1) for f in hidden]
    assert torch.allclose(model(text, *poses, *hidden), model(text, *poses, *noisy))
    with pytest.raises(ValueError, match="pre_face must have shape"):
        model(text, *poses)


def test_a_sequence_is_required():
    with pytest.raises(ValueError, match="leaves no sequence"):
        build_model(ModelConfig(8, 3, 3, model="text_gru", inputs="text"))


@pytest.mark.parametrize("cell", CELLS)
def test_baselines_train_and_evaluate(make_manifest, tmp_path, cell):
    context = {"context": np.array([1, 0, 1, 0], np.float32)}
    train, val, test = (
        make_manifest(split, overrides=context) for split in ("train", "val", "test")
    )
    output = tmp_path / "best.pt"
    main(
        ["--model", f"annotation_{cell}", "--train-manifest", str(train)]
        + ["--val-manifest", str(val), "--test-manifest", str(test)]
        + ["--num-joints", "3", "--joint-features", "3", "--width", "16"]
        + ["--epochs", "2", "--augment", "--output", str(output)]
    )
    checkpoint = torch.load(output, weights_only=True)
    assert checkpoint["config"]["model"] == f"annotation_{cell}"
    assert checkpoint["config"]["text_dim"] == 4
    report = json.loads(output.with_suffix(".metrics.json").read_text())
    assert report["test"]["aggregate"]["n"] == 2
