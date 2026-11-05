import pytest
import torch

from asd_contextpose import (
    AnnotationContextPose,
    ContextPose,
    ModelConfig,
    build_model,
)


def inputs(config, batch=2):
    return [torch.randn(batch, config.text_dim)] + [
        torch.randn(batch, config.steps, config.num_joints, config.joint_features)
        for _ in range(3)
    ]


@pytest.mark.parametrize("batch", [1, 2])
def test_forward_backward_and_shared_weights(config, batch):
    model = ContextPose(config)
    calls = []
    hook = model.pose_encoder.register_forward_hook(lambda *args: calls.append(1))
    pred = model(*inputs(config, batch))
    hook.remove()
    assert pred.shape == (batch, 2)
    assert torch.isfinite(pred).all()
    assert len(calls) == 3
    pred.square().sum().backward()
    for parameter in (
        model.text_projection.weight,
        model.pose_encoder[0].weight,
        model.region_embeddings,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert (model.region_embeddings.grad.abs().sum(dim=1) > 0).all()


@pytest.mark.parametrize(
    "bad", ["text", "time", "joint", "batch", "dtype", "nan", "empty"]
)
def test_input_validation(config, bad):
    values = inputs(config)
    if bad == "text":
        values[0] = values[0][:, :-1]
    elif bad == "time":
        values[1] = values[1][:, :-1]
    elif bad == "joint":
        values[2] = values[2][:, :, :-1]
    elif bad == "batch":
        values[3] = values[3][:1]
    elif bad == "dtype":
        values[1] = values[1].double()
    elif bad == "nan":
        values[3][0, 0, 0, 0] = float("nan")
    else:
        values = [value[:0] for value in values]
    with pytest.raises(ValueError):
        ContextPose(config)(*values)


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(8, 0, 3)


def test_one_batch_overfit(config):
    model = ContextPose(config).eval()  # Disable dropout for stable loss comparison.
    values = inputs(config)
    target = torch.tensor([[0.3, -0.2], [-0.4, 0.6]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial = (model(*values) - target).square().mean(dim=0).sum().item()
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(*values) - target).square().mean(dim=0).sum()
        loss.backward()
        optimizer.step()
    final = (model(*values) - target).square().mean(dim=0).sum().item()
    print(f"synthetic one-batch loss: {initial:.6f} -> {final:.6f}")
    assert final < initial * 0.2


def test_ablations_ignore_the_dropped_input(config):
    text, *poses = inputs(config)
    for ignored, kwargs in (("pose", {"inputs": "text"}), ("text", {"inputs": "pose"})):
        model = ContextPose(ModelConfig(8, 3, 3, dropout=0.0, **kwargs)).eval()
        changed = [torch.randn_like(p) for p in poses] if ignored == "pose" else poses
        other_text = torch.randn_like(text) if ignored == "text" else text
        torch.testing.assert_close(model(text, *poses), model(other_text, *changed))


def test_text_skip_and_width_change_the_model():
    model = ContextPose(ModelConfig(8, 3, 3, width=32, text_skip=True))
    assert model.text_projection.out_features == 32
    assert model(*inputs(model.config)).shape == (2, 2)
    with pytest.raises(ValueError):
        ModelConfig(8, 3, 3, width=30)
    with pytest.raises(ValueError):
        ModelConfig(8, 3, 3, inputs="video")


@pytest.mark.parametrize("variant", ["annotation_film", "annotation_attention"])
def test_annotation_models_forward_and_backward(variant):
    model = build_model(ModelConfig(4, 3, 3, model=variant))
    assert isinstance(model, AnnotationContextPose)
    context = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 1.0, 0], [0, 0, 0, 0]])
    poses = inputs(model.config, batch=3)[1:]
    pred = model(context, *poses)
    assert pred.shape == (3, 2)
    pred.square().sum().backward()
    conditioning = model.film if variant == "annotation_film" else model.text_projection
    assert all(p.grad is not None for p in conditioning.parameters())
    assert model.pose_encoder[0].weight.grad.abs().sum() > 0


def test_film_starts_as_identity_and_annotations_need_four_features():
    model = build_model(ModelConfig(4, 3, 3, model="annotation_film"))
    gamma_beta = model.film(torch.eye(4))
    assert (gamma_beta == 0).all()
    with pytest.raises(ValueError, match="4 context features"):
        ModelConfig(768, 3, 3, model="annotation_attention")
    with pytest.raises(ValueError, match="context must have shape"):
        model(torch.zeros(2, 5), *inputs(model.config)[1:])


def test_text_attention_is_the_default_model(config):
    model = build_model(config)
    assert type(model) is ContextPose and config.model == "text_attention"
    assert config.head_dropout == 0.0 and not hasattr(model, "film")


def faces(config, batch=2, mask=1.0):
    """Face tensors [B, steps, face_dim + 1]; the last column is the gate mask."""
    features = torch.randn(batch, config.steps, config.face_dim)
    gate = torch.full((batch, config.steps, 1), mask)
    return [torch.cat([features, gate], -1) for _ in range(3)]


def test_face_stream_starts_at_zero_and_corrects_valence_only():
    config = ModelConfig(8, 3, 3, dropout=0.0, face_dim=5)
    model = ContextPose(config).eval()
    values, face = inputs(config), faces(config)
    plain = ContextPose(ModelConfig(8, 3, 3, dropout=0.0))
    plain.load_state_dict(model.state_dict(), strict=False)
    torch.testing.assert_close(model(*values, *face), plain.eval()(*values))
    model.face_head[-1].weight.data.normal_()
    other = faces(config)
    a, b = model(*values, *face), model(*values, *other)
    assert not torch.allclose(a[:, 0], b[:, 0])
    torch.testing.assert_close(a[:, 1], b[:, 1])


def test_face_stream_ignores_masked_steps_and_faceless_samples():
    config = ModelConfig(8, 3, 3, dropout=0.0, face_dim=5, face_targets="both")
    model = ContextPose(config).eval()
    model.face_head[-1].weight.data.normal_()
    model.face_head[-1].bias.data.fill_(1.0)
    values, face = inputs(config), faces(config)
    for tensor in face:
        tensor[:, ::2, -1] = 0
    changed = [t.clone() for t in face]
    for tensor in changed:
        tensor[:, ::2, :-1] = torch.randn_like(tensor[:, ::2, :-1])
    torch.testing.assert_close(model(*values, *face), model(*values, *changed))
    with_face = model(*values, *face)
    no_face = model(*values, *faces(config, mask=0.0))
    torch.testing.assert_close(no_face - with_face, -model._face(face))


def test_face_only_input_and_face_validation():
    config = ModelConfig(8, 3, 3, face_dim=5, inputs="face")
    model = ContextPose(config)
    pred = model(*inputs(config), *faces(config))
    assert pred.shape == (2, 2) and (pred[:, 1] == 0).all()
    pred[:, 0].sum().backward()
    assert model.face_encoder[1].weight.grad is not None
    with pytest.raises(ValueError, match="face_dim > 0"):
        ModelConfig(8, 3, 3, inputs="face")
    with pytest.raises(ValueError, match="pre_face must have shape"):
        model(*inputs(config))
    with pytest.raises(ValueError, match="face inputs need"):
        ContextPose(ModelConfig(8, 3, 3))(*inputs(config), *faces(config))
    annotation = build_model(
        ModelConfig(4, 3, 3, model="annotation_attention", face_dim=5)
    )
    assert annotation(torch.zeros(2, 4), *inputs(config)[1:], *faces(config)).shape == (
        2,
        2,
    )
