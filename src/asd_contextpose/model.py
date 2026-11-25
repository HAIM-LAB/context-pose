"""External context conditions shared temporal pose attention.

The context is either a CLIP text embedding (`text_attention`) or the four
binary event annotations (`annotation_attention`, `annotation_film`). The
`*_gru` / `*_bilstm` models are recurrent baselines (`baseline_models`). An
optional face stream (`face_dim > 0`) pools frozen per-step face features over
each region and adds its own correction, to valence only by default.

The per-step subject features are the pose (`subject="pose"`), frozen
EmotionCLIP frame embeddings (`"eclip"`), both (`"both"`) or neither (`"none"`).
With both, `fusion` either projects each to half the width and concatenates
them before one temporal encoder (`"concat"`), or encodes each separately and
lets the context query attend over both token sequences (`"separate"`).

Further per-step feature streams (`streams`, e.g. Video Swin or HOG-HOF
features from `video_features`) each get their own temporal encoder and
modality embedding, and their tokens join the sequence the query attends over.
A stream named `vst` enters as `pre_vst`/`event_vst`/`post_vst` [B, steps,
dim + 1]: the features, then a 0/1 child-tracked flag, all used as features.
`stream_dropout` hides each stream of a training sample with that chance.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

INPUTS = ("all", "text", "pose", "face")
RECURRENT = ("gru", "bilstm")
MODELS = (
    "text_attention",
    "annotation_film",
    "annotation_attention",
    *(
        f"{condition}_{cell}"
        for condition in ("text", "annotation")
        for cell in RECURRENT
    ),
)
# social_interaction, auditory_event, activity_object_change, visual_media_event
CONTEXT_DIM = 4
FILM_HIDDEN = 64
FACE_HIDDEN = 32
FACE_TARGETS = ("valence", "both")
FACE_KEYS = ("pre_face", "event_face", "post_face")
ECLIP_KEYS = ("pre_eclip", "event_eclip", "post_eclip")
SUBJECTS = ("pose", "eclip", "both", "none")
REGIONS = ("pre", "event", "post")
RESERVED_STREAMS = ("pose", "face", "eclip")  # their keys are taken
FUSIONS = ("concat", "separate")
# "regression": outputs are [valence, arousal] on the label scale. "votes": the
# same outputs are log-odds of an annotator voting positive valence / high arousal.
# "final": log-odds that the final annotation is positive valence / high arousal.
OBJECTIVES = ("regression", "votes", "final")


@dataclass(frozen=True)
class ModelConfig:
    text_dim: int  # width of the conditioning vector (CONTEXT_DIM for annotations)
    num_joints: int
    joint_features: int
    steps: int = 32
    width: int = 128
    dropout: float = 0.1  # inside the temporal Transformer layer
    head_dropout: float = 0.0  # on the regression head input
    inputs: str = "all"  # ablations: "text" ignores pose, "pose" ignores context,
    # "face" uses only the face stream
    text_skip: bool = False  # also add the projected context to the head input
    model: str = "text_attention"
    face_dim: int = 0  # per-step face features (plus a mask column); 0 = no face
    face_targets: str = "valence"  # outputs the face stream corrects
    face_dropout: float = 0.0  # training-time chance of hiding a sample's face
    objective: str = "regression"
    subject: str = "pose"  # per-step subject features: pose, EmotionCLIP or both
    eclip_dim: int = 0  # EmotionCLIP embedding width (plus a tracked-flag column)
    fusion: str = "concat"  # how subject="both" combines the two
    streams: tuple[tuple[str, int], ...] = ()  # further (name, width) streams
    stream_dropout: float = 0.0  # training-time chance of hiding a sample's stream

    def __post_init__(self) -> None:
        # Checkpoints and JSON may hand back lists; keep the config hashable.
        streams = tuple(tuple(stream) for stream in self.streams)
        object.__setattr__(self, "streams", streams)
        for stream in streams:
            if (
                len(stream) != 2
                or type(stream[0]) is not str
                or not stream[0].isidentifier()
                or stream[0] in RESERVED_STREAMS
                or type(stream[1]) is not int
                or stream[1] < 1
            ):
                raise ValueError(
                    "streams must be (name, width) pairs with an identifier name "
                    f"other than {RESERVED_STREAMS} and a positive width"
                )
        if len({name for name, _ in streams}) != len(streams):
            raise ValueError("stream names must be distinct")
        for name in ("text_dim", "num_joints", "joint_features", "steps", "width"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % 4:
            raise ValueError("width must be divisible by the 4 attention heads")
        if type(self.face_dim) is not int or self.face_dim < 0:
            raise ValueError("face_dim must be a non-negative integer")
        if self.face_targets not in FACE_TARGETS:
            raise ValueError(f"face_targets must be one of {FACE_TARGETS}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        if self.inputs == "face" and not self.face_dim:
            raise ValueError("inputs='face' requires face_dim > 0")
        for name in ("dropout", "head_dropout", "face_dropout", "stream_dropout"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.inputs not in INPUTS:
            raise ValueError(f"inputs must be one of {INPUTS}")
        if type(self.text_skip) is not bool:
            raise ValueError("text_skip must be a bool")
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        if self.subject not in SUBJECTS:
            raise ValueError(f"subject must be one of {SUBJECTS}")
        if self.fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}")
        if type(self.eclip_dim) is not int or self.eclip_dim < 0:
            raise ValueError("eclip_dim must be a non-negative integer")
        if (self.subject in ("eclip", "both")) != (self.eclip_dim > 0):
            raise ValueError(
                "eclip_dim > 0 is required exactly when subject uses eclip"
            )
        if self.subject == "none" and not self.streams:
            raise ValueError("subject='none' requires at least one stream")
        if (self.subject != "pose" or self.streams) and self.model.endswith(RECURRENT):
            raise ValueError("recurrent baselines take pose subjects only")
        if self.condition_key == "context" and self.text_dim != CONTEXT_DIM:
            raise ValueError(f"annotation models take {CONTEXT_DIM} context features")

    @property
    def condition_key(self) -> str:
        """Sample key of the conditioning vector."""
        return "text_embedding" if self.model.startswith("text_") else "context"

    @property
    def stream_keys(self) -> dict[str, int]:
        """Sample key (e.g. `event_vst`) -> per-step width, flag column included."""
        return {
            f"{region}_{name}": dim + 1
            for name, dim in self.streams
            for region in REGIONS
        }


def check_inputs(
    c: ModelConfig,
    device: torch.device,
    condition: Tensor,
    poses: tuple[Tensor, Tensor, Tensor],
    faces: tuple[Tensor | None, Tensor | None, Tensor | None],
    eclips: tuple[Tensor | None, Tensor | None, Tensor | None] = (None,) * 3,
    streams: dict[str, Tensor] | None = None,
) -> None:
    """Raise ValueError unless the inputs match the config (shape, dtype, device)."""
    key = c.condition_key
    pre_pose, event_pose, post_pose = poses
    if condition.ndim != 2 or condition.shape[1] != c.text_dim:
        raise ValueError(f"{key} must have shape [B, {c.text_dim}]")
    batch = condition.shape[0]
    if batch == 0:
        raise ValueError("empty batches are not supported")
    expected = (batch, c.steps, c.num_joints, c.joint_features)
    for name, tensor in (
        (key, condition),
        ("pre_pose", pre_pose),
        ("event_pose", event_pose),
        ("post_pose", post_pose),
    ):
        if name != key and tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tensor.device != device:
            raise ValueError(f"{name} must be on the model device")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain finite values")
    if c.face_dim:
        face_shape = (batch, c.steps, c.face_dim + 1)
        for name, tensor in zip(FACE_KEYS, faces):
            if tensor is None or tuple(tensor.shape) != face_shape:
                raise ValueError(f"{name} must have shape {face_shape}")
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite float32")
    elif any(tensor is not None for tensor in faces):
        raise ValueError("face inputs need a model with face_dim > 0")
    if c.eclip_dim:
        eclip_shape = (batch, c.steps, c.eclip_dim + 1)
        for name, tensor in zip(ECLIP_KEYS, eclips):
            if tensor is None or tuple(tensor.shape) != eclip_shape:
                raise ValueError(f"{name} must have shape {eclip_shape}")
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite float32")
    elif any(tensor is not None for tensor in eclips):
        raise ValueError("eclip inputs need a model with eclip_dim > 0")
    streams = streams or {}
    if set(streams) != set(c.stream_keys):
        raise ValueError(
            f"stream inputs must be exactly {sorted(c.stream_keys)}, "
            f"got {sorted(streams)}"
        )
    for name, width in c.stream_keys.items():
        tensor = streams[name]
        if tuple(tensor.shape) != (batch, c.steps, width):
            raise ValueError(f"{name} must have shape {(batch, c.steps, width)}")
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite float32")
        if tensor.device != device:
            raise ValueError(f"{name} must be on the model device")


def temporal_encoder(channels: int, width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(channels, width, 3, padding=1),
        nn.GELU(),
        nn.Conv1d(width, width, 3, padding=1),
        nn.GELU(),
    )


class ContextPose(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        width = config.width
        pose_dim = config.num_joints * config.joint_features
        eclip_dim = config.eclip_dim + 1
        if config.subject == "both" and config.fusion == "concat":
            # Equal halves, so the 513-d embedding cannot swamp the 44 pose features.
            self.pose_projection = nn.Sequential(
                nn.LayerNorm(pose_dim), nn.Linear(pose_dim, width // 2)
            )
            self.eclip_projection = nn.Sequential(
                nn.LayerNorm(eclip_dim), nn.Linear(eclip_dim, width // 2)
            )
            self.subject_encoder = temporal_encoder(width, width)
        else:
            if config.subject in ("pose", "both"):
                self.pose_encoder = temporal_encoder(pose_dim, width)
            if config.subject in ("eclip", "both"):
                self.eclip_norm = nn.LayerNorm(eclip_dim)
                self.eclip_encoder = temporal_encoder(eclip_dim, width)
            if config.subject == "both":
                self.modality_embeddings = nn.Parameter(torch.empty(2, width))
                nn.init.normal_(self.modality_embeddings, std=0.02)
        if config.streams:
            self.stream_norms = nn.ModuleList(
                nn.LayerNorm(dim + 1) for _, dim in config.streams
            )
            self.stream_encoders = nn.ModuleList(
                temporal_encoder(dim + 1, width) for _, dim in config.streams
            )
            self.stream_embeddings = nn.Parameter(
                torch.empty(len(config.streams), width)
            )
            nn.init.normal_(self.stream_embeddings, std=0.02)
        self.text_projection = nn.Linear(config.text_dim, width)
        self.region_embeddings = nn.Parameter(torch.empty(3, width))
        nn.init.normal_(self.region_embeddings, std=0.02)
        if config.model == "annotation_film":
            # [gamma, beta] = f_c(c); zero-initialised so training starts at identity.
            self.film = nn.Sequential(
                nn.Linear(config.text_dim, FILM_HIDDEN),
                nn.GELU(),
                nn.Linear(FILM_HIDDEN, 2 * width),
            )
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        self.cross_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                width,
                4,
                dim_feedforward=2 * width,
                dropout=config.dropout,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.dropout = nn.Dropout(config.head_dropout)
        self.head = nn.Sequential(
            nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 2)
        )
        if config.face_dim:
            self.face_encoder = nn.Sequential(
                nn.LayerNorm(config.face_dim),
                nn.Linear(config.face_dim, FACE_HIDDEN),
                nn.GELU(),
            )
            outputs = 1 if config.face_targets == "valence" else 2
            self.face_head = nn.Sequential(
                nn.Linear(3 * FACE_HIDDEN, width // 2),
                nn.GELU(),
                nn.Linear(width // 2, outputs),
            )
            # Zero-initialised so training starts from the pose-only prediction.
            nn.init.zeros_(self.face_head[-1].weight)
            nn.init.zeros_(self.face_head[-1].bias)

    def forward(
        self,
        text_embedding: Tensor,
        pre_pose: Tensor,
        event_pose: Tensor,
        post_pose: Tensor,
        pre_face: Tensor | None = None,
        event_face: Tensor | None = None,
        post_face: Tensor | None = None,
        pre_eclip: Tensor | None = None,
        event_eclip: Tensor | None = None,
        post_eclip: Tensor | None = None,
        **streams: Tensor,
    ) -> Tensor:
        """Return unconstrained [valence, arousal] predictions, shape [B, 2].

        Face tensors are [B, steps, face_dim + 1]: features, then a 0/1 mask of
        the steps whose face passed the quality gate. EmotionCLIP and stream
        tensors (`streams`, keyed like `event_vst`) are [B, steps, dim + 1]:
        the features, then a 0/1 child-tracked flag, all used as features.
        """
        c = self.config
        faces = (pre_face, event_face, post_face)
        eclips = (pre_eclip, event_eclip, post_eclip)
        check_inputs(
            c,
            self.region_embeddings.device,
            text_embedding,
            (pre_pose, event_pose, post_pose),
            faces,
            eclips,
            streams,
        )
        extras = self._streams(streams)
        batch = text_embedding.shape[0]
        condition = text_embedding
        face = self._face(faces) if c.face_dim else 0.0
        if c.inputs == "face":
            return face
        text = self.text_projection(condition)
        if c.inputs == "text":
            return self.head(self.dropout(text)) + face
        if c.inputs == "pose":
            text = torch.zeros_like(text)
        film = c.model == "annotation_film"
        if film:
            context = torch.zeros_like(condition) if c.inputs == "pose" else condition
            gamma, beta = self.film(context).unsqueeze(1).chunk(2, dim=-1)
        regions = []
        for region, pose in enumerate((pre_pose, event_pose, post_pose)):
            tokens = self._tokens(pose.flatten(2), eclips[region], extras[region])
            if film:
                # Context modulates the pose features; the region embedding alone
                # queries them, so the context reaches the output only via FiLM.
                tokens = (1 + gamma) * tokens + beta
                query = self.region_embeddings[region].expand(batch, 1, -1)
            else:
                query = (text + self.region_embeddings[region]).unsqueeze(1)
            summary, _ = self.cross_attention(query, tokens, tokens, need_weights=False)
            regions.append(summary.squeeze(1))
        fused = self.temporal_encoder(torch.stack(regions, dim=1)).mean(dim=1)
        if c.text_skip:
            fused = fused + text
        return self.head(self.dropout(fused)) + face

    def _streams(self, streams: dict[str, Tensor]) -> list[list[Tensor]]:
        """Per region, the stream tensors in config order, hidden by stream dropout."""
        c = self.config
        names = [name for name, _ in c.streams]
        extras = [[streams[f"{r}_{name}"] for name in names] for r in REGIONS]
        if self.training and c.stream_dropout and names:
            batch = extras[0][0].shape[0]
            device = extras[0][0].device
            keep = torch.rand(batch, len(names), device=device) >= c.stream_dropout
            keep = keep.float()[:, :, None, None]
            extras = [[x * keep[:, i] for i, x in enumerate(r)] for r in extras]
        return extras

    def _tokens(self, pose: Tensor, eclip: Tensor | None, extras: list[Tensor]):
        """[B, n * steps, width] tokens: subject features, then each stream."""
        c = self.config

        def encode(encoder: nn.Module, x: Tensor) -> Tensor:
            return encoder(x.transpose(1, 2)).transpose(1, 2)

        if c.subject == "pose":
            parts = [encode(self.pose_encoder, pose)]
        elif c.subject == "eclip":
            parts = [encode(self.eclip_encoder, self.eclip_norm(eclip))]
        elif c.subject == "none":
            parts = []
        elif c.fusion == "concat":
            steps = torch.cat(
                [self.pose_projection(pose), self.eclip_projection(eclip)], -1
            )
            parts = [encode(self.subject_encoder, steps)]
        else:
            pose_tokens = encode(self.pose_encoder, pose) + self.modality_embeddings[0]
            eclip_tokens = encode(self.eclip_encoder, self.eclip_norm(eclip))
            parts = [pose_tokens, eclip_tokens + self.modality_embeddings[1]]
        for index, x in enumerate(extras):
            tokens = encode(self.stream_encoders[index], self.stream_norms[index](x))
            parts.append(tokens + self.stream_embeddings[index])
        return parts[0] if len(parts) == 1 else torch.cat(parts, 1)

    def _face(self, faces: tuple[Tensor, ...]) -> Tensor:
        """[B, 2] correction from the masked mean face features of each region."""
        c = self.config
        mask = torch.stack([face[..., -1] for face in faces], 1)  # [B, 3, steps]
        if self.training and c.face_dropout:
            hide = torch.rand(mask.shape[0], device=mask.device) < c.face_dropout
            mask = mask * (~hide).float()[:, None, None]
        pooled = []
        for region, face in enumerate(faces):
            weight = mask[:, region, :, None]
            hidden = self.face_encoder(face[..., :-1]) * weight
            pooled.append(hidden.sum(1) / weight.sum(1).clamp(min=1))
        delta = self.face_head(self.dropout(torch.cat(pooled, -1)))
        # No usable face in any region: no correction (not the head's bias).
        delta = delta * (mask.sum((1, 2)) > 0).float()[:, None]
        if c.face_targets == "valence":
            delta = nn.functional.pad(delta, (0, 1))
        return delta


class AnnotationContextPose(ContextPose):
    """ContextPose conditioned on binary event annotations instead of CLIP text."""

    def forward(
        self,
        context: Tensor,
        pre_pose: Tensor,
        event_pose: Tensor,
        post_pose: Tensor,
        pre_face: Tensor | None = None,
        event_face: Tensor | None = None,
        post_face: Tensor | None = None,
        pre_eclip: Tensor | None = None,
        event_eclip: Tensor | None = None,
        post_eclip: Tensor | None = None,
        **streams: Tensor,
    ) -> Tensor:
        return super().forward(
            context,
            pre_pose,
            event_pose,
            post_pose,
            pre_face,
            event_face,
            post_face,
            pre_eclip,
            event_eclip,
            post_eclip,
            **streams,
        )


def build_model(config: ModelConfig) -> nn.Module:
    if config.model.endswith(RECURRENT):
        from .baseline_models import build_baseline

        return build_baseline(config)
    if config.model == "text_attention":
        return ContextPose(config)
    return AnnotationContextPose(config)
