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

## Preparing the ViTPoseClean dataset

`dataset/` (git-ignored) holds `ViTPoseClean/{train,test}/*_ctx.npz` pose clips,
their `clean_qc.csv` files, and `Final_Annotation_v2.xlsx`. Each clip is
`[5 s pre-event] + [event] + [5 s post-event]`. Convert it with:

ViTPoseClean loses the child in 19% of frames, mostly through tracking/selection
rather than detection, which empties the pre, event, or post window of 43 clips.
`gapfill` recovers these frames from `dataset/full_video/{split}/<clip>.mp4` into
`dataset/ViTPoseFilled` (same schema plus a `filled` mask). Starting at the nearest
frame ViTPoseClean kept, it detects people (RT-DETR), estimates their pose
(ViTPose-base), and accepts the person closest to the previous pose, in shoulder
widths, trying rotated frames when nobody matches; it stops following after
0.5 s without a match. Upstream frames are left untouched. On frames both cover,
its poses agree with ViTPoseClean to a median 0.04 shoulder widths. It is
resumable; `--device cuda` is much faster than the CPU default.

`gapfill --reacquire` is an optional second pass, run on clips that still have
an empty window, for gaps too long to follow into. It recognises the child by
the colour of their clothing below the shoulders, learned against the other
people in the clip. Clips where that colour does not tell the child apart
(threshold above 0.7) are left alone: on review, the two clips above it had
locked onto a different child. New frames are marked `reacquired` (with
`seeded` start points) in each file. Spot-check them with
`dataset/visual_checks/reacquired_check.py` before training.

```bash
uv run python -m asd_contextpose.gapfill            # ViTPoseClean -> ViTPoseFilled
uv run python -m asd_contextpose.gapfill --reacquire --clips ...  # optional, see above
uv run python -m asd_contextpose.prepare            # add --require-qc to drop QC failures
uv run python -m asd_contextpose.train \
  --train-manifest data/vitpose/train.csv --val-manifest data/vitpose/val.csv \
  --text-dim 768 --num-joints 11 --joint-features 3 --device cuda \
  --output checkpoints/best.pt
```

The converter deviates from the defaults above in documented ways:

- It reads `dataset/ViTPoseFilled` by default (`--poses ViTPoseClean` for the
  unfilled poses); manifests record each clip's `pose_filled_frac`.

- Coordinates are the dataset's `keypoints_norm` (origin at the shoulder midpoint,
  unit = the clip's median shoulder width), not image-relative; features are
  `[x, y, confidence]` with confidence clipped to `[0, 1]`.
- Gaps the upstream cleaner left unfilled stay zero with confidence zero rather
  than being bridged. Clips with no event frames (≤ 10 s) or with a fully
  unobserved region are excluded and listed on stdout.
- `text_embedding` is the frozen, L2-normalised CLIP ViT-L/14 text embedding
  (`openai/clip-vit-large-patch14`, `text_dim=768`) of each clip's sentence in
  `ASD_ContextPose_Main_Event_Text_Natural.csv`; the weights download from the
  Hugging Face Hub on first use. Manifests record the pinned model revision and the
  sentence. Those sentences are generated from the consensus indicators, so for an
  indicator-only baseline use `--text-encoder indicators` (`text_dim=5`, in the
  order listed under the input contract).
- Every NPZ also holds `context`: the four binary indicators `social_interaction`,
  `auditory_event`, `activity_object_change`, `visual_media_event` (all zero means
  no event), used by the annotation models below.
- `val` holds out whole training videos (`--val-fraction`, stratified by category).
  No participant IDs exist, so `participant_id` is the video ID; the same child
  may appear in more than one video, so splits are video-independent only.
  Videos `109_v_ArmFlapping_59` and `183_v_HandAction_05_0` are the same footage
  filed under two categories; `DUPLICATE_VIDEOS` maps them to one participant
  and recording so no split separates them.
- Labels are confidence-weighted averages of three annotators' votes (valence:
  positive/neutral/negative; arousal: high/low only). Every NPZ also holds
  `votes` `[2]`, the share of known votes for positive valence and for high
  arousal, and manifests keep each annotator's vote as `valence_votes` /
  `arousal_votes` (`1|0|` style; empty = unknown).
- `needs_review` comes from `Final_detailed` (the `Final` sheet omits it).
- `--speed` adds a fourth pose feature, joint speed in shoulder widths per second,
  measured at 25 fps and averaged over each resampling step (resampling alone
  would scale speed by the window length).

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
| `text_embedding` | `[D_text]` (`text_attention` model) |
| `context` | `[4]` binary annotations (`annotation_*` models) |
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

### Choosing a model

`--model` selects how the environmental context conditions the pose branch.
All three share the pose encoder, temporal fusion, regression head, loss and
metrics; only the context path differs.

| `--model` | Context input | Conditioning |
| --- | --- | --- |
| `text_attention` (default) | `text_embedding`, CLIP `[768]` | Projected text plus a region embedding queries each region's pose tokens. |
| `annotation_attention` | `context`, binary `[4]` | Same attention, with the projected annotations as the query: `q_r = W_c c + e_r`. |
| `annotation_film` | `context`, binary `[4]` | `[γ, β] = f_c(c)` (4 → 64 → 2×128, zero-initialised) rescales each region's pose features, `(1 + γ) ⊙ Z_r + β`; the region embedding alone then queries them. |

The annotation models never read CLIP embeddings, and `--text-dim` defaults to 4
for them:

```bash
uv run python -m asd_contextpose.train --model annotation_film \
  --train-manifest data/vitpose/train.csv --val-manifest data/vitpose/val.csv \
  --num-joints 11 --joint-features 3 --device cuda --output checkpoints/film.pt
```

`--model` works the same way with `crossval`. `--inputs pose` gives a pose-only
baseline that ignores the context, and `--inputs text` a context-only one.

### Face stream

Pose drops the face, which is where valence mostly shows. The optional face
stream adds frozen facial-expression features beside the pose branch:

```bash
uv run python -m asd_contextpose.face --device cuda   # -> dataset/FaceFeatures
uv run python -m asd_contextpose.prepare --speed --face dataset/FaceFeatures   --output data/vitpose_face            # --face-features probs for the 7 classes
uv run python -m asd_contextpose.crossval --face-dim 768 --face-dropout 0.3 ...
```

- `face` crops the child's face at each of the 32 resampling steps per region
  (centred on the tracked face keypoints, 0.9 shoulder widths wide) and embeds it
  with a frozen FER2013 ViT (`trpakov/vit-face-expression`): the 768-d `[CLS]`
  and the 7 expression probabilities, with quality measures. OpenCV's YuNet face
  detector (downloaded once to `~/.cache/asd_contextpose`) scores each step on a
  wider crop: pose keypoints alone report eyes on lowered or hair-covered heads.
- `prepare --face` writes `pre_face`/`event_face`/`post_face` `[32, D + 1]`: the
  features, then a 0/1 quality gate (nose and both eyes tracked, YuNet score at
  least 0.8 at the tracked face, crop not dark, not blurred). Failed steps are zeroed and masked; manifests record each clip's
  `face_kept_frac`.
- The model encodes each step (`LayerNorm → Linear(D, 32) → GELU`), averages the
  kept steps per region, and maps the three region means to a correction added to
  the prediction: valence only by default (`--face-targets both` for both).
  The correction starts at zero, is zero for clips with no kept face, and
  `--face-dropout` hides a training clip's face to keep the pose path working.
  `--inputs face` uses the face stream alone (with `--residual-by`, on top of the
  lookup).

The FER model was trained on adult faces. On these clips it is reliable for
"happy" on clear, frontal faces; it reads lowered or hair-covered heads as "sad"
and talking or wide laughs as angry/surprise/fear. Check crops with
`dataset/visual_checks/face_crops_check.py` (local only: children's faces).

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

With few participants, a single validation split is too noisy to compare
settings. `crossval` runs participant-grouped k-fold cross-validation over the
train and val manifests (never test). Each fold is evaluated once, by a model
early-stopped on a different fold. It reports pooled out-of-fold metrics against
an always-predict-the-training-median baseline, with a participant-bootstrap
95% interval for the difference. Other options are forwarded to `train`:

```bash
uv run python -m asd_contextpose.crossval \
  --manifests data/vitpose/train.csv data/vitpose/val.csv --folds 5 \
  --text-dim 768 --num-joints 11 --joint-features 3 --device cuda \
  --output checkpoints/cv            # writes checkpoints/cv/cv_report.json
```

### Annotator votes and binary calls

Annotators agree only moderately (Fleiss' kappa about 0.33 for both targets),
and the weighted consensus splits identical vote patterns across class
boundaries: a "neutral" arousal label only arises from disagreement. Two
binary calls follow the votes instead: positive vs not-positive valence and
high vs low arousal, by annotator majority (ties left out).

`--objective votes` trains on the vote shares directly: the two outputs become
log-odds of an annotator voting positive valence / high arousal, the loss is
cross-entropy against the shares, early stopping uses validation cross-entropy,
and `--residual-by` starts from the log-odds of each group's mean share.

When the manifests carry votes, `crossval` reports both calls for any model in
`cv_report.json` under `classification`: AUC, accuracy and balanced accuracy,
with the decision threshold chosen per fold on the clips the model was fitted
and selected on. Each is compared with always predicting the majority class, a
lookup of the mean vote share per `--baseline-by` value, and the annotators
themselves (each against the other two where those agree). Prefer balanced
accuracy and AUC: the classes are unbalanced (most clips are positive valence
and low arousal), so plain accuracy rewards always predicting the majority.

## Recommended model (V3)

Frozen after cross-validation: context + face, trained on annotator votes.
Valence is read mostly from the face and arousal mostly from the event context;
pose did not help.

```bash
uv run python -m asd_contextpose.face --device cuda   # FER embeddings, YuNet scores
uv run python -m asd_contextpose.prepare --speed --face dataset/FaceFeatures \
  --output data/vitpose_face_det
V3="--num-joints 11 --joint-features 4 --epochs 60 --patience 10 \
  --model annotation_attention --lr 3e-4 --width 64 --dropout 0.3 \
  --head-dropout 0.3 --weight-decay 0.05 --augment --objective votes \
  --face-dim 768 --face-dropout 0.3 --inputs face --face-targets both \
  --residual-by event_text"
for seed in 1 2 3 4 5; do                               # final 5-seed ensemble
  uv run python -m asd_contextpose.train --device cuda --seed $seed --text-dim 4 \
    --train-manifest data/vitpose_face_det/train.csv \
    --val-manifest data/vitpose_face_det/val.csv $V3 \
    --output checkpoints/final_v3/seed$seed.pt
done
```

Each checkpoint stores its binary-call rules (`classification`): decision
thresholds fitted on its own training and validation clips. `--evaluate` takes
several checkpoints as an ensemble (mean score for AUC, majority of the
members' calls) and, on clips with annotator votes, reports both calls against
always predicting the majority, the event-context lookup, and the annotators:

```bash
uv run python -m asd_contextpose.train --evaluate checkpoints/final_v3/seed*.pt \
  --test-manifest <new labelled clips>.csv --output checkpoints/final_v3/new.json
```

Cross-validated results (153 development clips, 5 participant-grouped folds,
5-seed ensemble, participant-bootstrap 95% intervals):

| Binary call | Predictor | Accuracy | Balanced accuracy | AUC |
| --- | --- | --- | --- | --- |
| Valence: positive vs not (100 of 150 positive) | Annotators (each vs the other two) | 79% | 79% | - |
| | Always majority | 67% | 50% | 0.50 |
| | Event-context lookup | 55% | 53% | 0.55 |
| | **V3** | 67% | **65%** [53, 75] | **0.71** [0.56, 0.83] |
| Arousal: high vs low (33 of 148 high) | Annotators (each vs the other two) | 81% | 77% | - |
| | Always majority | 78% | 50% | 0.50 |
| | Event-context lookup | 60% | 68% | 0.63 |
| | **V3** | 68% | **69%** [57, 79] | **0.73** [0.62, 0.85] |

V3 is well above chance on both calls, but its gain over the lookup is not
significant (balanced accuracy +12% [-3, +25] for valence, +1% [-9, +9] for
arousal), it was the best of six configurations compared, and plain accuracy
does not beat always predicting the majority. Confirm it on new labelled
clips: the test split has already been used once, for an earlier pose
candidate. Face embeddings partly encode identity and groups are videos, so
the same child may appear in several folds; participant IDs would rule out
identity leakage.

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
