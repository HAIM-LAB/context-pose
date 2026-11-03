# ASD-ContextPose

A minimal supervised PyTorch model for continuous **[valence, arousal]**
regression from an upstream text embedding and three pose-keypoint sequences.
The architecture shares a two-layer TCN across regions, adds learned pre/event/post
embeddings to projected text queries, applies shared four-head cross-attention,
and combines the three summaries with one Transformer layer and an MLP.
This is a proposed adaptation inspired by EmotionCLIP, not a reproduction.

## Setup and verification

This existing `cvpr` project retains its Python 3.13 requirement. Install with uv:

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run pytest -q -s tests/test_model.py::test_one_batch_overfit
```

The lockfile includes PyTorch, NumPy, pytest, and Ruff; no pretrained downloads
are needed. On Windows and Linux, `pyproject.toml` pins torch to the PyTorch CUDA
13.2 wheel index (`cu132`, which includes Blackwell `sm_120` kernels and needs an
NVIDIA driver supporting CUDA ≥ 13.2); macOS uses the PyPI build. To target a
different CUDA version, change the `pytorch-cu132` index URL and run `uv lock`.
Check the install with
`uv run python -c "import torch; print(torch.cuda.is_available())"`, then pass
`--device cuda` to training (the CLI still defaults to CPU).
Tests generate synthetic tensors only in temporary directories. They verify
optimization and plumbing, not scientific predictive performance.

## Input contract

```python
from asd_contextpose import ContextPose, ModelConfig

# Set these dimensions from the upstream embedding and pose specifications.
model = ContextPose(ModelConfig(text_dim=D_text, num_joints=J, joint_features=F))
pred = model(text_embedding, pre_pose, event_pose, post_pose)  # [B, 2]
```

All inputs must be finite float32 tensors on the model's device. Text is
`[B,D_text]`; each pose region is `[B,32,J,F]`. Predictions are unconstrained;
no sigmoid/tanh assumes an unconfirmed label range. Labels never enter forward.

Agree with the text-encoder developer on the embedding model/version, dimension,
normalization, and the mapping from event labels to faithful descriptions. Supply
the resulting embedding directly; this package does not generate text, look up
event labels, run CLIP, decode video, or extract poses. The five upstream indicators
are `social_interaction`, `auditory_event`, `activity_object_change`,
`visual_media_event`, and `no_event`; upstream must make `no_event` exclusive of
the other indicators. Embeddings of label combinations do not add unseen event
details. The text describes the event, not an event occurring throughout pre/post.

Preprocess every example upstream using one documented protocol:

- Select the same participant and fixed ordered joint list for all regions.
  Use the 5 seconds before the event, its actual interval, and the 5 seconds after.
- For the default image-relative coordinate protocol, divide x by image width
  and y by image height; use feature order `[x,y]` or `[x,y,confidence]` with
  confidence in `[0,1]`. Never supply raw pixel coordinates. If a different
  normalization is agreed upon, document it and use it consistently across splits;
  any learned normalization statistics must be fitted on training participants only.
- Interpolate missing coordinates in time between valid observations within the
  same region. For otherwise missing joints, zero-fill normalized x/y and use
  confidence zero; this fallback requires the confidence channel (`F=3`). With
  `F=2`, exclude samples with unresolved missing joints. Never interpolate across
  examples, subjects, or split boundaries. Fully unobserved regions should be
  excluded upstream; this MVP has no missing-region mask.
- Linearly interpolate each region on 32 equally spaced time points between its
  first and last observed timestamps; repeat a single valid frame if necessary.
  Interpolate confidence too, keeping missing positions at confidence zero.
  A 32-step event interval can represent a different duration from either flank.
  `--steps` allows a deliberate common-T preprocessing experiment.
- Preserve actual event start/end/duration, normalization version, joint order,
  and annotation protocol alongside the samples. Review the target definition
  before deciding whether post-event inputs are permissible.

The loader checks tensor structure and finite values, not the semantic correctness
of coordinate normalization, embeddings, labels, or inferred joint identities.

## Manifests and leakage checks

Use one CSV per split with required columns:

```csv
path,participant_id,recording_id,split
examples/window_001.npz,participant_001,recording_001,train
```

Paths are relative to the CSV (absolute paths also work). Split values are exactly
`train`, `val`, or `test`, matching the file's role. Each NPZ must contain:

| Key | float32 shape |
| --- | --- |
| `text_embedding` | `[D_text]` |
| `pre_pose`, `event_pose`, `post_pose` | each `[32,J,F]` |
| `target` | `[2]`, ordered valence then arousal |

NPZ files are loaded with `allow_pickle=False`. All eligible samples are validated
before training. Optional manifest fields are retained in `dataset.rows`:

- `needs_review`: `true`/`false` or `1`/`0`; true rows are excluded before NPZ loading.
  Only mark false after adjudication. Absent means the exporter has already
  excluded unadjudicated samples.
- `joint_order`: a `|`-separated list of J distinct joint names. If provided, it
  must match for all eligible samples and across splits/checkpoints. If omitted,
  the exporter is responsible for verifying joint ordering; array shape alone
  cannot prove it. Prefer including it for real datasets.
- `event_start,event_end,event_duration`: true event timing in seconds on the
  recording clock. Supplied endpoints and duration are validated for consistency,
  retained as metadata, and never used as model inputs.
- `post_start,post_end,outcome_start,outcome_end`: actual retained post-input
  interval and a self-stimulation outcome interval, when applicable to the target.
  Both endpoints of any supplied interval are required. Outcome timing requires
  post timing; overlapping half-open intervals are rejected. Truncate and resample
  upstream, or exclude the example, and document the changed prediction task.
  If timing is absent, an upstream leakage audit is required; the loader cannot
  infer temporal leakage from keypoint values.

Participants, globally unique recording IDs, and resolved sample paths must be
disjoint across train/validation/test. Assign whole recordings (including all
overlapping windows) to one split, and use stable participant IDs across recordings.
These checks rely on correct provenance IDs. The loader validates supplied splits;
it does not randomly split windows. With too few participants for a credible
held-out evaluation, use separate participant-grouped cross-validation manifests
and checkpoint/output paths for each fold, and report that limitation. Never relax
the group checks or describe a random-window split as participant independent.

## Training and evaluation

Set the three shell variables below to the dimensions confirmed with upstream:

```bash
uv run python -m asd_contextpose.train \
  --train-manifest data/train.csv --val-manifest data/val.csv \
  --text-dim "$D_TEXT" --num-joints "$J" --joint-features "$F" \
  --epochs 30 --batch-size 16 --lr 0.001 --patience 5 --seed 42 \
  --output checkpoints/best.pt
```

Training uses AdamW and the sum of per-target MSEs. Early stopping selects the
lowest validation mean MAE, with five consecutive non-improving epochs by default.
Seeds and deterministic Torch operations support reproducibility on the same
software/hardware; results need not be identical across devices or releases.
`--device cuda` or `--device mps` can be requested on supported installations.
The best checkpoint stores weights, dimensions, seed, epoch, validation results,
joint order, and train/validation group membership. Treat it as research data
because it includes participant/recording IDs and local sample paths.

`best.metrics.json` contains training history, separate valence/arousal MAE, RMSE,
CCC, their mean MAE, and participant-level metrics. Aggregate metrics weight
examples equally; they are not participant-balanced averages. CCC uses population
moments; it is JSON `null` for singleton groups or zero-denominator constant series.
Add `--test-manifest data/test.csv` to evaluate the best checkpoint once after
selection, or evaluate a saved checkpoint separately:

```bash
uv run python -m asd_contextpose.train \
  --evaluate checkpoints/best.pt --test-manifest data/test.csv \
  --output checkpoints/test.metrics.json
```

Standalone evaluation reloads dimensions and rejects groups used for training or
validation. Do not tune on the test set. No real training dataset is included.

## Interpretation and scope

Environmental labels and observable pose do not directly measure internal emotion
or establish causation. Targets must be interpreted under their annotation
protocol. With future/post-event observations this model characterizes a completed
window; it is not a prospective predictor unless every input precedes the defined
target. Attention weights are not causal explanations.

Pose-only, event-only, and simple-concatenation baselines are needed to test whether
context conditioning adds predictive information. They, ST-GCN, and VA-guided
contrastive learning are deferred beyond this supervised MVP. No clinical or
scientific performance claims follow from the synthetic smoke tests.
