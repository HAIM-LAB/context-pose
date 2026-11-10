# ASD-ContextPose: Methodology

This document describes the data, features, models, training and evaluation built in this project so far. The results themselves, with full tables, are in [`asd_contextpose_report.tex`](asd_contextpose_report.tex). The submission tables are in [`test.tex`](test.tex).

**Contents**

1. [Task](#1-task)
2. [Data](#2-data)
3. [Labels](#3-labels)
4. [Input streams and features](#4-input-streams-and-features)
5. [Models](#5-models)
6. [Objectives](#6-objectives)
7. [Training](#7-training)
8. [Evaluation protocol](#8-evaluation-protocol)
9. [Experimental programme](#9-experimental-programme)
10. [Summary of what the method has established](#10-summary-of-what-the-method-has-established)
11. [Limitations of the method](#11-limitations-of-the-method)
12. [Reproduction and code map](#12-reproduction-and-code-map)

---

## 1. Task

Each clip shows one child around an environmental event. It is split into three windows:

- **pre:** the 5 s before the event;
- **event:** the event itself;
- **post:** the 5 s after the event.

Three annotators rated each clip for:

- the child's **valence** (positive / neutral / negative);
- the child's **arousal** (high / low);
- four binary **context indicators** describing the event: social interaction, auditory event, activity or object change, visual or media event.

The goal is to predict the annotators' valence and arousal from what can be observed: the event context and the child's pose, face and scene.

---

## 2. Data

### 2.1 Source data

| Property | Value |
|---|---|
| Context clips | 192 (159 train, 33 test) |
| Source videos | 92 (71 train, 21 test) |
| Behaviour categories | ArmFlapping, HandAction, HeadBanging, Spinning |
| Clip length | 10–62 s (median 22 s) |
| Pose source | Upstream ViTPose pipeline (`ViTPoseClean`) |
| Joints | 11 upper-body COCO joints: nose, eyes, ears, shoulders, elbows, wrists |
| Frame rate | 25 fps |
| Pose normalisation | Centred on the shoulder midpoint, scaled by the clip's median shoulder width |

### 2.2 Audit

Before any modelling, all sources were cross-checked: the pose files, the QC tables and the annotation workbook. They were internally consistent:

- Clip names, splits and categories agree across all sources.
- No video crosses the train/test split.
- The context labels equal the annotators' majority vote.
- The valence and arousal targets equal the workbook's confidence-weighted consensus formula.
- The natural-language event sentences are generated from the four indicators. So there are only 10 distinct sentences in the dataset (9 among the development clips).

The audit also found four problems, each handled as follows:

| Problem | Handling |
|---|---|
| 62 clips (32%) fail the pose cleaner's own QC | Missing poses recovered by gap filling (§2.3) |
| The `needs_review` flag exists only in the `Final_detailed` sheet (3 clips) | Flagged clips are excluded |
| No participant IDs | Groups are videos; the same child may appear in several videos (§11) |
| `109_v_ArmFlapping_59` and `183_v_HandAction_05_0` are the same footage under two categories | Merged into one participant and recording (`DUPLICATE_VIDEOS`) |

### 2.3 Recovering missing poses (gap filling)

In `ViTPoseClean` the child has no pose in 19% of frames. This leaves at least one of the three windows empty in 43 clips (22%).

The losses fall unevenly across categories:

| Category | Clips affected |
|---|---|
| HeadBanging | 50% |
| Spinning | 37% |
| ArmFlapping | 12% |
| HandAction | 0% |

Dropping these clips would therefore bias the data. Visual review showed that most empty windows were tracking failures, not detection failures: the child was visible.

`gapfill.py` recovers the missing frames from the source videos.

**Pass 1, detection and tracking:**

- RT-DETR person detection plus ViTPose-base pose estimation.
- It steps outward from the nearest tracked frame.
- It accepts the detected person closest to the previous pose, measured in shoulder widths.
- It tries rotated frames when nobody matches.
- It stops after 0.5 s without a match.
- On frames both pipelines cover, its poses agree with the upstream ones to a median of 0.04 shoulder widths.

**Pass 2, re-acquisition (`--reacquire`):**

- It re-finds the child after long gaps by the colour of the clothing below the shoulders.
- The colour model is learned against the other people in the clip.
- Clips whose colour threshold exceeds 0.7 are skipped. The two clips above that threshold had locked onto a different child.

| Stage | Frames without pose | Clips with an empty window | Usable clips |
|---|---|---|---|
| `ViTPoseClean` | 21,026 (19%) | 43 (22.4%) | 148 |
| Gap fill, pass 1 | 10,815 (10%) | 9 (4.7%) | 182 |
| Pass 2 (re-acquisition) | – | 6 (3.1%) | **185** |

Seven clips remain excluded:

- four have a window that cannot be recovered;
- two need a human decision on which child is the subject;
- one is exactly 10 s long, which leaves no event frames.

### 2.4 Splits

The splits come from `prepare.py`:

- The validation split holds out whole training videos, stratified by category.
- The prepared data has 123 train, 31 validation and 31 test clips.
- The **development set** is train + validation: 154 clips from 68 videos. Of these, 153 are usable after removing one `needs_review` clip. All model selection uses cross-validation on these 153 clips (§8).
- The **test set** has 29 usable clips from 20 videos. It has been used exactly once (§8.5).

---

## 3. Labels

### 3.1 Consensus labels and votes

Each annotator votes as follows:

- **Valence:** positive / neutral / negative, coded 1 / 0.5 / 0.
- **Arousal:** high or low only.

Two label forms are stored per clip:

- **`target`:** the workbook's confidence-weighted consensus valence and arousal, on [−1, 1]. Used in the regression phase.
- **`votes`:** the share of known annotator votes for positive valence and for high arousal. Used by the votes objective (§6).

### 3.2 Annotator agreement

Agreement is moderate:

| Measure | Valence | Arousal |
|---|---|---|
| Fleiss' κ | 0.34 | 0.33 |
| All three annotators agree | 52% | 60% |
| An annotator matches the other two, where those two agree | 77% | 82% |
| Always guessing the majority class | 74% | 80% |

The labels are also unbalanced. Most clips are positive valence and low arousal, and only 7 development clips are negative valence.

### 3.3 From three classes to binary calls

The project began with three-class targets: the consensus label cut at ±1/3. Under that scheme, every model's accuracy was at or below always predicting the majority class (73% valence, 69% arousal).

The analysis found four causes:

1. **Annotator noise sets a low ceiling.** A single annotator beats the majority guess by only 2–3 points.
2. **The cut points do not match the annotation scheme.**
   - Arousal votes are only high or low, so the "neutral arousal" class (24 clips) is really annotator disagreement.
   - The valence vote pattern "two neutral, one positive" became neutral in 5 clips and positive in 11, depending only on confidence weights.
   - 16–17% of labels lie within 0.1 of a cut point.
3. **The inputs rarely make a minority class the likelier one.**
   - In 8 of 9 context combinations, the majority class is also the most common.
   - The regression models' minority-class predictions were right only 13–43% of the time.
4. **Classifiers trained for accuracy collapse to the majority.** Unweighted logistic classifiers made 0–14 minority predictions out of 153.

The labels were therefore redefined as the annotators' own **binary calls**, by annotator majority with ties excluded:

| Call | Clips | Positive class |
|---|---|---|
| Valence: positive vs not positive | 150 | 100 positive |
| Arousal: high vs low | 148 | 33 high |

Results are reported as balanced accuracy and AUC against the annotator ceiling (§8.3). Models are trained on the vote shares (§6).

---

## 4. Input streams and features

Each clip is converted by `prepare.py` into one NPZ file. Features are organised into four **streams**, named by the letters used throughout the results:

| Stream | Letter | Content |
|---|---|---|
| Pose | **P** | Per-window joint trajectories, optionally with real-time speed |
| Event annotation | **A** | The four binary context indicators (or their CLIP sentence embedding) |
| VLM scene description | **V** | A 384-d embedding of a vision-language model's description of the clip |
| Face | **F** | Per-step facial-expression embeddings with a quality mask |

Further frozen features were also tested: EmotionCLIP, a second (uploaded) face model, V-JEPA 2 video, and audio.

### 4.1 Pose (P)

`pre_pose`, `event_pose` and `post_pose` each have shape [32, 11, F].

- Each window is linearly resampled to **32 steps**, whatever its real length (5 to over 30 s).
- The base features are [x, y, confidence].
- **Speed channel (`--speed`)**, F = 4:
  - A fourth feature: joint speed in shoulder widths per second, measured at 25 fps.
  - It is capped at 20, to suppress tracking jumps.
  - It is averaged over each resampling bin.
  - Without this channel, frame-to-frame change in the resampled input no longer reflects real speed. With it, speed keeps real-time units whatever the window length.

### 4.2 Event annotation (A)

- `context`: the four binary indicators, shape [4]. All zero means no event.
- `text_embedding`: the frozen, L2-normalised CLIP ViT-L/14 embedding (768-d) of the event sentence.

Because the sentences are generated from the indicators, both forms carry the same 9-way information. They performed identically (§9.1).

### 4.3 VLM scene description (V)

This feature set was provided to the project:

- Qwen3-Omni descriptions of 5–6 s windows, covering the scene, the child's actions and the audible context.
- Each description is embedded with `all-MiniLM-L6-v2` (384-d).
- The embeddings are averaged into one L2-normalised clip vector.

The VLM embedding is stored as the clip's `text_embedding` and used as the network's context vector *c*. Depending on the offset (§5.2), it either replaces the annotation or is used alongside it.

The descriptions mention visible affect ("laughing" in 34 clips, "smiling" in 29, "crying" in 14). So part of the VLM signal is the VLM's own reading of the child's expression.

Data copies:

- `data/vitpose_v3vlm`: VLM `text_embedding` with pose.
- `data/vitpose_face_vlm`: VLM `text_embedding`, FER face and pose with speed.

### 4.4 Face (F)

`face.py` builds the face stream:

1. **Crop.** At each of the 32 steps per window, crop the child's face.
   - The crop is centred on the tracked face keypoints and is 0.9 shoulder widths wide.
   - It is resized to 224 px.
2. **Embed.** Pass the crop through the frozen FER ViT `trpakov/vit-face-expression` and keep the 768-d [CLS] embedding.
3. **Quality gate.** Visual checks showed the FER model is reliable for "happy" on clear frontal faces. It reads lowered or hair-covered heads as "sad" (the top class in 48% of clips) and dark crops as "angry". A step is therefore kept only if all of the following hold:
   - the face is frontal: the nose and both eyes are tracked;
   - OpenCV's YuNet face detector scores at least 0.8 at the tracked face location. This threshold was set by inspecting crops sorted by score, without labels;
   - the crop's brightness is at least 40;
   - the crop's sharpness (Laplacian variance) is at least 10.

The gate keeps 39% of steps. 27 development clips have no usable face.

`pre_face`, `event_face` and `post_face` each have shape [32, D+1]: the per-step features, followed by the 0/1 gate mask.

### 4.5 Other frozen features

**EmotionCLIP** (CVPR 2023; ViT-B/32 with a subject mask and a temporal transformer), extracted by a scratchpad script:

- **Crop:** a square of 0.875 × the frame's short side, centred on the child rather than the frame, resized to 224 px.
- **Subject mask:** the box around the child's tracked upper-body joints, padded and extended down to cover the torso. Untracked steps use an empty mask. The child is tracked in 94% of steps.
- **Frame embeddings:** one 512-d embedding (`encode_image`) at each of the 32 steps per window.
- **Window embedding:** one 512-d embedding (`encode_video`) per window, from 8 evenly spaced steps, following the authors' 8-frame protocol.
- Both embeddings are L2-normalised.

**Uploaded ViT face features (`face_features_v3/vit`)** were provided to the project:

- Per-frame arrays: `backbone_cls`, `backbone`, `eq_embed`, `eq_hidden` and `contrast`.
- They come with the provider's own face detection, which finds a face in 64% of frames.
- We used `backbone_cls` at the frame nearest each pose step, masked by `face_found`.
- Their cosine similarity with our FER embedding on matched frames is 0.17.

**Media features**, tested only as linear probes (§9.3):

- V-JEPA 2 on a child crop;
- AST audio embeddings;
- AudioSet event probabilities plus loudness.

**Movement features,** used in a linear probe: nine hand-crafted features per window:

- head, elbow, wrist and body speed;
- wrist height and spread;
- head bob.

---

## 5. Models

All networks output two unconstrained values ŷ ∈ ℝ², for valence and arousal. They share the same data loader, training loop, objectives and evaluation.

Notation:

- *P_r* ∈ ℝ^{32×J×F} is the pose of window *r* ∈ {pre, event, post}.
- *c* is the context vector.
- *W* is the model width.

### 5.1 ContextPose: context-queried temporal attention (`model.py`)

**Pose branch**, shared across the three windows:

```
Z_r = TCN(P_r) ∈ ℝ^{32×W}             TCN: Conv1d(J·F → W, k=3) → GELU → Conv1d(W → W, k=3) → GELU
q_r = W_c c + e_r                     e_r: learned region embedding (pre / event / post)
s_r = MHA(q_r, Z_r, Z_r)              4-head cross-attention: the context-dependent query attends over the 32 steps
h   = mean_r Transformer([s_pre, s_event, s_post])
                                      1 encoder layer, 4 heads, feed-forward width 2W, dropout
ŷ   = MLP(dropout(h)) + Δ_face + b_g   MLP: W → W/2 → GELU → 2
```

The context therefore decides which moments of each window the model attends to. A single transformer layer then relates the three windows.

**Conditioning variants** (`--model`):

| Variant | Context *c* | How context enters |
|---|---|---|
| `text_attention` | CLIP sentence embedding (768-d), or the VLM embedding (384-d, `--text-dim 384`) | Through the attention query |
| `annotation_attention` | The 4 binary indicators | Through the attention query |
| `annotation_film` | The 4 binary indicators | Through FiLM only: [γ, β] = f_c(c) (4 → 64 → 2W, zero-initialised) modulates the pose features, Z′_r = (1+γ) ⊙ Z_r + β, and the region embedding alone is the query |

There is also a **text-skip** option, which adds the projected context to the fused representation before the head: h + W_c c. It was used in one VLM + pose run.

**Stream ablation modes** (`--inputs`):

| Mode | Prediction | Streams used |
|---|---|---|
| `all` | MLP(h) + Δ_face + b_g | Pose, context, face (if configured) |
| `pose` | As `all`, with W_c c set to zero | Pose, face (if configured) |
| `text` | MLP(W_c c) + Δ_face + b_g | Context, face (if configured); no pose |
| `face` | Δ_face + b_g | Face only (plus the offset) |

### 5.2 Residual learning on the lookup (`--residual-by`)

The network predicts a correction to a fixed per-group offset fitted on its own training folds:

ŷ = f(x) + b_{g(x)}

- **`--residual-by event_text`:** the groups are the event-indicator combinations, so *b_g* is the **context lookup**:
  - the group median label for the regression objective;
  - the log-odds of the group's mean vote share for the votes objective.
  
  Groups with fewer than 3 training clips use the overall value. The network starts from the strongest baseline instead of having to re-learn it.
- **`--residual-by joint_order`:** this column is the same for every clip, so *b_g* is a constant fitted on the training folds. No annotation information reaches the model through the offset. This is how the "no annotation" rows are built.

The offset is how the event annotation (A) enters a model when the VLM embedding takes the context slot:

| Context vector *c* | Offset | Streams |
|---|---|---|
| Indicators | `event_text` | A |
| Indicators | `joint_order` | None from A, except through *c* in `all` / `text` modes |
| VLM embedding | `joint_order` | V **instead of** A |
| VLM embedding | `event_text` | A (through the offset) **and** V (through *c*) |

### 5.3 Face stream

The face stream sits beside the pose branch:

```
u_{r,t} = GELU(Linear_{D→32}(LayerNorm(x_{r,t})))       per-step face encoding
ū_r     = Σ_t m_{r,t} u_{r,t} / max(1, Σ_t m_{r,t})     masked mean over gated steps, per window
Δ_face  = g([ū_pre; ū_event; ū_post])                    g: 96 → W/2 → GELU → 1 or 2 outputs
```

- The last layer of *g* is **zero-initialised**, so training starts from the face-free prediction.
- Δ_face = 0 for clips with no gated face step in any window, rather than the head's bias.
- **Face dropout** (p = 0.3) hides a training clip's whole face stream.
- **`--face-targets`** sets whether the face corrects valence only or both targets.

**V3**, the frozen valence candidate, is `annotation_attention` with:

- `--inputs face --face-targets both`;
- `--residual-by event_text`;
- the votes objective.

Its prediction is therefore the context lookup's log-odds plus a learned face correction for both targets. The pose branch is unused. Five final checkpoints, each with its own decision thresholds, are frozen in `checkpoints/final_v3`.

### 5.4 Recurrent baselines: GRU and BiLSTM (`baseline_models.py`)

The baselines read the three windows in time order as **one 96-step sequence** (pre ‖ event ‖ post).

**Each step** is built as follows:

1. The step vector is [region one-hot (3); pose (J·F); face encoding (32) × mask; mask].
2. It is projected: Linear → W, then GELU and dropout.

**Recurrence:**

- GRU with W units, or BiLSTM with W/2 units per direction.
- The summary *h* is the GRU's last hidden state, or the BiLSTM's last forward and first backward states.

**Head:** MLP([h; W_c c]), 2W → W/2 → 2, plus *b_g*.

- Variants: `annotation_gru` and `annotation_bilstm` (indicators as *c*), `text_gru` and `text_bilstm` (VLM as *c*).
- The `--inputs`, `--residual-by`, objective and training flags behave as in ContextPose:
  - `text` reads the face sequence with the context, and no pose;
  - `face` reads the face sequence alone, with the context zeroed;
  - `pose` reads the pose sequence (with the face sequence when `--face-dim` is set) and zeroes the context;
  - face dropout (p = 0.3) hides a clip's whole face sequence, as in ContextPose.
- Unlike ContextPose they use no attention, no region pooling and no gated averaging of the face. The recurrence must summarise the clip itself, including learning to ignore masked face steps.
- Configurations with no sequence to read (annotation only, annotation + VLM) have no recurrent counterpart.

### 5.5 Linear probes

Frozen features were also tested with linear models in the same folds.

**Regression:**

- Ridge or L1-penalised linear regression, fitted as a correction to the training folds' lookup.

**Binary calls:**

- Class-balanced logistic regression on:
  - the one-hot indicator combination;
  - the FER face embedding (with a flag for clips without a usable face);
  - the EmotionCLIP window embeddings;
  
  as each configuration lists.
- Each embedding is standardised and reduced by PCA to 8 or 16 components, fitted in-fold. 32 components was also tested.
- The regularisation strength C ∈ {0.01, 0.1, 1, 10} is chosen by log-loss on an inner fold: the next fold, held out from the training folds.
- The whole procedure is repeated over 5 fold arrangements (fold seeds), because a single arrangement is noisy.

---

## 6. Objectives

| Objective | Output | Loss | Early stopping |
|---|---|---|---|
| **Regression** (phase 1) | Consensus valence and arousal | Sum of per-target MSE, or L1 | Validation mean MAE |
| **Votes** (`--objective votes`, phase 2) | Log-odds *z* that an annotator votes positive valence / high arousal | Binary cross-entropy against the vote share *s*: −[s log σ(z) + (1−s) log(1−σ(z))] | Validation cross-entropy |

The votes objective trains directly on what the annotators said. This avoids the confidence-weighted average, which splits identical vote patterns across class boundaries (§3.3).

For binary calls, the model's score is σ(z). A call is made by thresholding the score (§8.3).

---

## 7. Training

All models use AdamW with batch size 16.

| Setting | Default | Regularised (used from phase 1 onward) |
|---|---|---|
| Width *W* | 128 | 64 |
| Dropout / head dropout | – | 0.3 / 0.3 |
| Weight decay | – | 0.05 |
| Learning rate | 1e-3 | 3e-4 |
| Pose augmentation | – | Rotation ±15°, scale 0.9–1.1, jitter σ = 0.02, left/right mirror (p = 0.5) |
| Epochs / patience | 30 / 5 | Phase 2: up to 60 / 10 |

**Epoch caps:**

- The VLM runs and the recurrent runs use a 150-epoch cap with patience 10.
- The 150-epoch cap for the lookup + VLM model was chosen after 5 of 25 models were still improving at epoch 60. That is a mild selection on the development clips.

The **"V3 settings"** used across phase 2 are the regularised settings with:

- `--num-joints 11 --joint-features 4` (pose with speed);
- the votes objective;
- for face models: `--face-dim 768 --face-dropout 0.3`.

---

## 8. Evaluation protocol

### 8.1 Cross-validation

A single 18–31-clip validation split is too noisy to compare models: one clip moves validation MAE by about 5%. All comparisons therefore use **grouped 5-fold cross-validation over the 153 development clips** (`crossval.py`):

- **Grouping:** videos are dealt to folds round-robin within each behaviour category, so all clips of a video share a fold. The duplicate videos are merged into one group.
- **Train / stop / score:** each fold is predicted once, by a model trained on three folds and early-stopped on the fourth. No clip is scored by a model selected on it.
- **Refitting:** the lookup offset, the thresholds and any PCA are all refitted inside each fold.
- **Seeds:** every network configuration is run with 5 training seeds.

**Fold arrangements by phase:**

- Phase 1 (regression) used the original folds, in which the duplicate videos fell into different folds.
- Phase 2 (binary calls) uses deduplicated folds.
- Numbers are comparable within a phase, not across phases.

### 8.2 Reference predictors

| Reference | Definition |
|---|---|
| **Overall median / always majority** | Ignores the inputs |
| **Context lookup** | For each of the 9 indicator combinations that occur: the training clips' median label (regression) or mean vote share (votes). Combinations with fewer than 3 training clips use the overall value. Rebuilt in every fold from that fold's training clips. |
| **Annotator ceiling** | Each annotator scored against the other two, on clips where those two agree |

### 8.3 Metrics

**Phase 1, regression:**

- MAE and concordance correlation coefficient (CCC) per target.
- The mean MAE over the two targets.

**Phase 2, binary calls:**

- **AUC** of the score.
- **Balanced accuracy:** the mean per-class recall, where chance is 50%. This is the headline metric, because 78% of clips are low arousal and 67% are positive valence.
- **Plain accuracy**, for comparison with always predicting the majority class.

**Thresholds:**

- The decision threshold maximises balanced accuracy on the clips each model was fitted and early-stopped on. It is never set on the scored fold.

**Ensembles:**

- A 5-seed ensemble takes the majority of the seeds' calls.
- Its score, for AUC, is the mean of the seeds' scores.

### 8.4 Uncertainty

- 95% intervals come from a **participant (video) bootstrap**: 2,000 resamples of whole groups.
- They are reported for balanced accuracy, AUC, and the **gain over the lookup** (model balanced accuracy minus lookup balanced accuracy, or lookup MAE minus model MAE).
- A gain is called significant only if its interval excludes zero.

### 8.5 Held-out test

The test split (29 clips, 20 videos) was used **exactly once**:

- It scored a candidate fixed in advance from cross-validation: the lookup for valence, and a linear context + movement model for arousal.
- Every later candidate is judged on cross-validation only.
- Every later candidate needs new labelled clips to confirm it.
- `train --evaluate` scores a checkpoint ensemble on new labelled clips against the majority, the lookup and the annotators.

---

## 9. Experimental programme

The experiments ran in two phases. Each step was motivated by the previous one.

### 9.1 Phase 1: regression

1. **Architecture comparison.** `text_attention` (CLIP), `annotation_attention`, `annotation_film` and pose only were each run with default and regularised training. They were compared with the overall median and the context lookup.
2. **Residual learning, L1 loss and speed.**
   - Residual learning on the lookup.
   - L1 vs MSE loss.
   - The real-time speed channel.
   - These were run singly and combined (configurations S1–S6).
3. **Linear movement probe.** Nine movement features per window plus context. This produced the candidate scored once on the test set.

### 9.2 Media and face features

1. **Frozen-feature probe.** FER embedding, FER expression probabilities, V-JEPA 2, AST and AudioSet features, each as an L1 correction on the lookup.
2. **Face stream inside the network**, with the quality gate:
   - F0: residual + speed, no face;
   - F1: face corrects valence;
   - F2: face corrects both targets;
   - F3: lookup + face, no pose.
3. **Face only**, from the overall median.

### 9.3 Phase 2: binary calls

1. **Objective comparison.** Regression- vs votes-trained models on the same inputs: pose + context, context + face, face only. This identified **V3**.
2. **EmotionCLIP.** Linear probes (EmotionCLIP alone; annotation + face + EmotionCLIP) and a network with an EmotionCLIP stream in place of the face stream.
3. **VLM and the uploaded face model.**
   - Lookup + VLM at 60- and 150-epoch caps.
   - Lookup + VLM with each face model.
   - Lookup + VLM + pose with speed, with and without text-skip.
   - VLM only.
4. **Stream ablation.** ContextPose is run on every combination of P, A, V and F:
   - 4 single streams;
   - 6 pairs;
   - 4 triples;
   - all four.
   
   Each uses the same folds, the votes objective and V3 settings (60-epoch cap). The "full model" is pose + annotation + face.

| Streams | `--model` | `--inputs` | Face flags | `--residual-by` |
|---|---|---|---|---|
| P | `annotation_attention` | `pose` | – | `joint_order` |
| A | `annotation_attention` | `text` | – | `event_text` |
| F | `annotation_attention` | `face` | yes | `joint_order` |
| P + A | `annotation_attention` | `all` | – | `event_text` |
| P + F | `annotation_attention` | `pose` | yes | `joint_order` |
| A + F (V3) | `annotation_attention` | `face` | yes | `event_text` |
| P + A + F (full) | `annotation_attention` | `all` | yes | `event_text` |
| V | `text_attention --text-dim 384` | `text` | – | `joint_order` |
| P + V | `text_attention --text-dim 384` | `all` | – | `joint_order` |
| V + F | `text_attention --text-dim 384` | `text` | yes | `joint_order` |
| A + V | `text_attention --text-dim 384` | `text` | – | `event_text` |
| P + V + F | `text_attention --text-dim 384` | `all` | yes | `joint_order` |
| P + A + V | `text_attention --text-dim 384` | `all` | – | `event_text` |
| A + V + F | `text_attention --text-dim 384` | `text` | yes | `event_text` |
| P + A + V + F | `text_attention --text-dim 384` | `all` | yes | `event_text` |

Face flags: `--face-dim 768 --face-dropout 0.3 --face-targets both`.

5. **Recurrent baselines.** GRU and BiLSTM on every stream combination that has a sequence, with the same protocol. Their results are compared cell by cell with ContextPose.

---

## 10. Summary of what the method has established

The full numbers are in the report. Briefly:

- **The context lookup is a strong baseline.** No regression network beat it. It is the only predictor confirmed on held-out videos.
- **Valence comes from the face.**
  - The best valence model is V3 (lookup + face): 65% balanced accuracy, AUC 0.71.
  - The annotator ceiling is 79%.
  - The lookup alone is at chance for valence (AUC 0.55).
- **Arousal comes from the context**, from either the annotation or the VLM.
  - Lookup + VLM ranks best: AUC 0.77.
  - VLM only makes the best calls: 76% balanced accuracy, against a ceiling of 77%.
  - A linear annotation + face + EmotionCLIP model reaches AUC 0.80–0.81.
- **The VLM description can replace the manual event annotation.** VLM + face roughly matches V3 on both targets.
- **Pose, including real-time speed, did not help.**
  - It is at chance alone.
  - Adding it never improved a ContextPose stream combination.
  - Its cross-validated arousal signal failed on the test set.
- **ContextPose beat the GRU/BiLSTM baselines.** No recurrent baseline reached the best ContextPose result for either target.
- **No gain over the lookup is significant.** All the intervals include zero.

---

## 11. Limitations of the method

- **Grouping is by video, not by child.** There are no participant IDs, and the same child may appear in several videos. The face, EmotionCLIP and VLM features can encode identity or room, so part of their gain may be memorisation.
- **Selection on the development folds.** The best configurations were chosen from many compared on the same folds, so their numbers are optimistic. The test split has been used once and must not be reused for selection.
- **Small, noisy labels.**
  - 153 development clips.
  - Moderate annotator agreement (κ ≈ 0.33).
  - Only 7 negative-valence and 33 high-arousal clips.
- **Pretrained models were trained on adults** (FER2013, EmotionCLIP's vloggers).
- **Face coverage is limited:** the gate leaves 27 clips with no usable face.
- **Some components live outside the package.** The EmotionCLIP extractor and the VLM / uploaded-ViT data builders are scratchpad scripts, not yet `prepare.py` options.
- **Licensing.** The EmotionCLIP weights have no stated licence, and `face_features_v3` is undocumented.
- **Scope.**
  - The labels are annotators' judgements of observed behaviour, not measurements of internal state.
  - The post-event window means the models characterise a completed episode rather than predict ahead.

---

## 12. Reproduction and code map

### 12.1 Pipeline

```bash
# 1. Face crops, FER embeddings and YuNet scores
uv run python -m asd_contextpose.face --device cuda

# 2. Convert clips to model inputs (pose with speed, gated face, votes)
uv run python -m asd_contextpose.prepare --speed --face dataset/FaceFeatures \
  --output data/vitpose_face_det

# 3. Cross-validate one configuration (one seed; repeat for seeds 1-5)
V3="--num-joints 11 --joint-features 4 --epochs 60 --patience 10 \
  --model annotation_attention --lr 3e-4 --width 64 --dropout 0.3 \
  --head-dropout 0.3 --weight-decay 0.05 --augment --objective votes \
  --face-dim 768 --face-dropout 0.3 --inputs face --face-targets both \
  --residual-by event_text"
uv run python -m asd_contextpose.crossval --device cuda --seed 1 \
  --manifests data/vitpose_face_det/train.csv data/vitpose_face_det/val.csv $V3 \
  --output checkpoints/votes/V3_context_face/seed1

# 4. Score a frozen ensemble on new labelled clips
uv run python -m asd_contextpose.train --evaluate checkpoints/final_v3/seed*.pt \
  --test-manifest <new clips>.csv --output checkpoints/final_v3/new.json
```

The VLM rows use `data/vitpose_face_vlm` with `--model text_attention --text-dim 384`. The recurrent baselines use `--model annotation_gru | annotation_bilstm | text_gru | text_bilstm`.

### 12.2 Code map (`src/asd_contextpose/`)

| Module | Role |
|---|---|
| `gapfill.py` | Pose recovery from the source videos (detection, tracking, clothing-colour re-acquisition) |
| `prepare.py` | Conversion to NPZ, speed channel, face gate, vote shares, splits |
| `face.py` | Face crops, FER embeddings, YuNet scores, brightness and sharpness |
| `data.py` | Dataset and loader |
| `model.py` | ContextPose, the annotation and FiLM variants, the face stream |
| `baseline_models.py` | GRU and BiLSTM baselines |
| `train.py` | Objectives, residual offsets, training loop, ensemble evaluation |
| `crossval.py` | Grouped 5-fold cross-validation |
| `votes.py` | Annotator votes, binary calls, thresholds, metrics, bootstrap |
| `linear_probe.py` | Movement features and linear probes |

The checkpoints and per-fold reports are under `checkpoints/`:

- `checkpoints/votes/`, `checkpoints/streams/`, `checkpoints/v3vlm*/`, `checkpoints/baselines/`;
- `checkpoints/final_v3/`, the frozen V3 ensemble.
