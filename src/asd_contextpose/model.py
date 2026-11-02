"""External text conditions shared temporal pose attention."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    text_dim: int
    num_joints: int
    joint_features: int
    steps: int = 32

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class ContextPose(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.pose_encoder = nn.Sequential(
            nn.Conv1d(config.num_joints * config.joint_features, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.GELU(),
        )
        self.text_projection = nn.Linear(config.text_dim, 128)
        self.region_embeddings = nn.Parameter(torch.empty(3, 128))
        nn.init.normal_(self.region_embeddings, std=0.02)
        self.cross_attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 4, dim_feedforward=256, batch_first=True),
            num_layers=1,
        )
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(
        self,
        text_embedding: Tensor,
        pre_pose: Tensor,
        event_pose: Tensor,
        post_pose: Tensor,
    ) -> Tensor:
        """Return unconstrained [valence, arousal] predictions, shape [B, 2]."""
        c = self.config
        if text_embedding.ndim != 2 or text_embedding.shape[1] != c.text_dim:
            raise ValueError(f"text_embedding must have shape [B, {c.text_dim}]")
        batch = text_embedding.shape[0]
        if batch == 0:
            raise ValueError("empty batches are not supported")
        expected = (batch, c.steps, c.num_joints, c.joint_features)
        for name, tensor in (
            ("text_embedding", text_embedding),
            ("pre_pose", pre_pose),
            ("event_pose", event_pose),
            ("post_pose", post_pose),
        ):
            if name != "text_embedding" and tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if tensor.dtype != torch.float32:
                raise ValueError(f"{name} must be float32")
            if tensor.device != self.region_embeddings.device:
                raise ValueError(f"{name} must be on the model device")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain finite values")

        text = self.text_projection(text_embedding)
        regions = []
        for region, pose in enumerate((pre_pose, event_pose, post_pose)):
            tokens = self.pose_encoder(pose.flatten(2).transpose(1, 2)).transpose(1, 2)
            query = (text + self.region_embeddings[region]).unsqueeze(1)
            summary, _ = self.cross_attention(query, tokens, tokens, need_weights=False)
            regions.append(summary.squeeze(1))
        fused = self.temporal_encoder(torch.stack(regions, dim=1)).mean(dim=1)
        return self.head(fused)
