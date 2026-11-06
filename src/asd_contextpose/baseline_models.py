"""Recurrent baselines: a GRU or a BiLSTM reads the clip's steps in order.

The pre-event, event and post-event regions are joined into one sequence of
3 x steps, each step tagged with its region (one-hot). A step holds the
flattened pose [joints x features] and, with a face stream, the projected face
features times the quality-gate mask, then the mask. The recurrent summary (the
GRU's last hidden state; the BiLSTM's last forward and first backward states)
is joined with the projected context and mapped to [valence, arousal].

Inputs, ablations and objectives are those of `model.ContextPose`, so training
and cross-validation run these unchanged (`--model annotation_gru`, ...):
"all" reads pose (+ face) with context, "pose" zeroes the context, "text" reads
the face sequence with context (no pose), "face" reads the face sequence alone.
"""

import torch
from torch import Tensor, nn

from .model import FACE_HIDDEN, RECURRENT, ModelConfig, check_inputs

REGIONS = 3


class RecurrentBaseline(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        cell = config.model.rsplit("_", 1)[1]
        if cell not in RECURRENT:
            raise ValueError(f"recurrent baselines are {RECURRENT}, not {cell}")
        self.use_pose = config.inputs in ("all", "pose")
        if not self.use_pose and not config.face_dim:
            raise ValueError(
                f"inputs={config.inputs!r} leaves no sequence: add face_dim"
            )
        self.config = config
        width = config.width
        step_width = REGIONS
        if self.use_pose:
            step_width += config.num_joints * config.joint_features
        if config.face_dim:
            self.face_encoder = nn.Sequential(
                nn.LayerNorm(config.face_dim),
                nn.Linear(config.face_dim, FACE_HIDDEN),
                nn.GELU(),
            )
            step_width += FACE_HIDDEN + 1
        self.step_projection = nn.Sequential(nn.Linear(step_width, width), nn.GELU())
        self.step_dropout = nn.Dropout(config.dropout)
        if cell == "gru":
            self.rnn = nn.GRU(width, width, batch_first=True)
        else:  # two directions of width / 2: the same summary width as the GRU
            self.rnn = nn.LSTM(width, width // 2, batch_first=True, bidirectional=True)
        self.text_projection = nn.Linear(config.text_dim, width)
        self.dropout = nn.Dropout(config.head_dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * width, width // 2), nn.GELU(), nn.Linear(width // 2, 2)
        )

    def forward(
        self,
        text_embedding: Tensor,
        pre_pose: Tensor,
        event_pose: Tensor,
        post_pose: Tensor,
        pre_face: Tensor | None = None,
        event_face: Tensor | None = None,
        post_face: Tensor | None = None,
    ) -> Tensor:
        """Return unconstrained [valence, arousal] predictions, shape [B, 2]."""
        c = self.config
        faces = (pre_face, event_face, post_face)
        check_inputs(
            c,
            self.text_projection.weight.device,
            text_embedding,
            (pre_pose, event_pose, post_pose),
            faces,
        )
        batch, steps = pre_pose.shape[:2]
        region = torch.eye(REGIONS, device=pre_pose.device).repeat_interleave(steps, 0)
        parts = [region.expand(batch, -1, -1)]
        if self.use_pose:
            parts.append(torch.cat((pre_pose, event_pose, post_pose), 1).flatten(2))
        if c.face_dim:
            face = torch.cat(faces, 1)
            mask = face[..., -1:]
            if self.training and c.face_dropout:
                hide = torch.rand(batch, device=mask.device) < c.face_dropout
                mask = mask * (~hide).float()[:, None, None]
            parts += [self.face_encoder(face[..., :-1]) * mask, mask]
        sequence = self.step_dropout(self.step_projection(torch.cat(parts, -1)))
        _, state = self.rnn(sequence)
        hidden = state[0] if isinstance(self.rnn, nn.LSTM) else state
        summary = torch.cat(tuple(hidden), -1)  # [B, width]: all layers/directions
        text = self.text_projection(text_embedding)
        if c.inputs in ("pose", "face"):
            text = torch.zeros_like(text)
        return self.head(self.dropout(torch.cat((summary, text), -1)))


class AnnotationRecurrentBaseline(RecurrentBaseline):
    """RecurrentBaseline conditioned on binary event annotations instead of text."""

    def forward(
        self,
        context: Tensor,
        pre_pose: Tensor,
        event_pose: Tensor,
        post_pose: Tensor,
        pre_face: Tensor | None = None,
        event_face: Tensor | None = None,
        post_face: Tensor | None = None,
    ) -> Tensor:
        return super().forward(
            context, pre_pose, event_pose, post_pose, pre_face, event_face, post_face
        )


def build_baseline(config: ModelConfig) -> RecurrentBaseline:
    if config.condition_key == "context":
        return AnnotationRecurrentBaseline(config)
    return RecurrentBaseline(config)
