import pytest
import torch

from asd_contextpose import ContextPose, ModelConfig


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
