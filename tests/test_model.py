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


def eclips(config, batch=2):
    """EmotionCLIP tensors [B, steps, eclip_dim + 1]; the last column is a flag."""
    return {
        f"{region}_eclip": torch.randn(batch, config.steps, config.eclip_dim + 1)
        for region in ("pre", "event", "post")
    }


@pytest.mark.parametrize(
    "subject, fusion", [("eclip", "concat"), ("both", "concat"), ("both", "separate")]
)
def test_eclip_subjects_forward_backward_and_use_their_inputs(subject, fusion):
    config = ModelConfig(
        8, 3, 3, dropout=0.0, subject=subject, eclip_dim=6, fusion=fusion
    )
    model = ContextPose(config).eval()
    values, extra = inputs(config), eclips(config)
    pred = model(*values, **extra)
    assert pred.shape == (2, 2)
    pred.sum().backward()
    other = {key: torch.randn_like(value) for key, value in extra.items()}
    assert not torch.allclose(pred, model(*values, **other))
    moved = [values[0], *(torch.randn_like(v) for v in values[1:])]
    uses_pose = not torch.allclose(pred, model(*moved, **extra))
    assert uses_pose == (subject == "both")
    annotation = build_model(
        ModelConfig(4, 3, 3, model="annotation_attention", subject=subject, eclip_dim=6)
    )
    assert annotation(torch.zeros(2, 4), *values[1:], **extra).shape == (2, 2)


def streams(config, batch=2):
    """Stream tensors [B, steps, dim + 1] keyed like `event_vst`."""
    return {
        key: torch.randn(batch, config.steps, width)
        for key, width in config.stream_keys.items()
    }


@pytest.mark.parametrize("subject", ["pose", "eclip", "both", "none"])
def test_streams_forward_backward_and_use_their_inputs(subject):
    config = ModelConfig(
        8,
        3,
        3,
        dropout=0.0,
        subject=subject,
        eclip_dim=6 if subject in ("eclip", "both") else 0,
        fusion="separate",
        streams=(("vst", 5), ("hoghof", 4)),
    )
    model = ContextPose(config).eval()
    values = inputs(config)
    extra = {**(eclips(config) if config.eclip_dim else {}), **streams(config)}
    pred = model(*values, **extra)
    assert pred.shape == (2, 2)
    pred.sum().backward()
    for index in range(2):
        assert model.stream_encoders[index][0].weight.grad.abs().sum() > 0
    assert (model.stream_embeddings.grad.abs().sum(1) > 0).all()
    for name in ("vst", "hoghof"):
        changed = {
            k: torch.randn_like(v) if k.endswith(name) else v for k, v in extra.items()
        }
        assert not torch.allclose(pred, model(*values, **changed))
    moved = [values[0], *(torch.randn_like(v) for v in values[1:])]
    uses_pose = not torch.allclose(pred, model(*moved, **extra))
    assert uses_pose == (subject in ("pose", "both"))
    annotation = build_model(
        ModelConfig(
            4,
            3,
            3,
            model="annotation_attention",
            subject=subject,
            eclip_dim=config.eclip_dim,
            streams=config.streams,
        )
    )
    assert annotation(torch.zeros(2, 4), *values[1:], **extra).shape == (2, 2)


def test_stream_dropout_hides_whole_streams_in_training_only():
    config = ModelConfig(
        8, 3, 3, dropout=0.0, subject="none", streams=(("vst", 5),), stream_dropout=0.5
    )
    model = ContextPose(config)
    values, extra = inputs(config, batch=64), streams(config, batch=64)
    torch.manual_seed(0)
    hidden = model.train()(
        *values, **{k: torch.zeros_like(v) for k, v in extra.items()}
    )
    torch.manual_seed(0)
    pred = model(*values, **extra)
    dropped = torch.isclose(pred, hidden).all(1)
    assert 0 < dropped.sum() < 64  # hidden samples match an all-zero stream
    model.eval()
    assert torch.allclose(model(*values, **extra), model(*values, **extra))
    assert not torch.isclose(model(*values, **extra), hidden).all(1).any()


def test_streams_keep_existing_models_unchanged():
    """No streams: the same parameters, so seeded runs and checkpoints still match."""
    base = ModelConfig(8, 3, 3, subject="both", eclip_dim=6, fusion="separate")
    torch.manual_seed(0)
    old = ContextPose(base)
    torch.manual_seed(0)
    new = ContextPose(ModelConfig(**{**base.__dict__, "streams": ()}))
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key])
    assert not any(k.startswith("stream") for k in old.state_dict())


def test_stream_validation():
    with pytest.raises(ValueError, match="requires at least one stream"):
        ModelConfig(8, 3, 3, subject="none")
    with pytest.raises(ValueError, match="eclip_dim > 0"):
        ModelConfig(8, 3, 3, subject="none", eclip_dim=6, streams=(("vst", 5),))
    for bad in ((("eclip", 5),), (("vst", 0),), (("v st", 5),), (("vst", 5.0),)):
        with pytest.raises(ValueError, match="streams must be"):
            ModelConfig(8, 3, 3, streams=bad)
    with pytest.raises(ValueError, match="distinct"):
        ModelConfig(8, 3, 3, streams=(("vst", 5), ("vst", 4)))
    with pytest.raises(ValueError, match="stream_dropout"):
        ModelConfig(8, 3, 3, streams=(("vst", 5),), stream_dropout=1.0)
    with pytest.raises(ValueError, match="pose subjects only"):
        ModelConfig(8, 3, 3, model="text_gru", streams=(("vst", 5),))
    listed = ModelConfig(8, 3, 3, streams=[["vst", 5]])  # as a checkpoint may hold
    assert listed.streams == (("vst", 5),) and hash(listed)
    config = ModelConfig(8, 3, 3, streams=(("vst", 5),))
    with pytest.raises(ValueError, match="stream inputs must be exactly"):
        ContextPose(config)(*inputs(config))
    with pytest.raises(ValueError, match="stream inputs must be exactly"):
        ContextPose(ModelConfig(8, 3, 3))(*inputs(config), **streams(config))
    wrong = {k: v[..., :-1] for k, v in streams(config).items()}
    with pytest.raises(ValueError, match="pre_vst must have shape"):
        ContextPose(config)(*inputs(config), **wrong)


def test_eclip_validation():
    with pytest.raises(ValueError, match="eclip_dim > 0"):
        ModelConfig(8, 3, 3, subject="both")
    with pytest.raises(ValueError, match="eclip_dim > 0"):
        ModelConfig(8, 3, 3, eclip_dim=6)
    with pytest.raises(ValueError, match="subject must be"):
        ModelConfig(8, 3, 3, subject="video", eclip_dim=6)
    with pytest.raises(ValueError, match="fusion must be"):
        ModelConfig(8, 3, 3, subject="both", eclip_dim=6, fusion="sum")
    with pytest.raises(ValueError, match="pose subjects only"):
        ModelConfig(8, 3, 3, model="text_gru", subject="both", eclip_dim=6)
    config = ModelConfig(8, 3, 3, subject="both", eclip_dim=6)
    with pytest.raises(ValueError, match="pre_eclip must have shape"):
        ContextPose(config)(*inputs(config))
    with pytest.raises(ValueError, match="eclip inputs need"):
        ContextPose(ModelConfig(8, 3, 3))(*inputs(config), **eclips(config))
