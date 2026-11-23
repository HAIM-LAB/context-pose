# Feature Engineering for a Language-Assisted Video Swin Transformer

Plan for evaluating the architecture of Deng et al., *Language-Assisted Deep Learning for Autistic Behaviors Recognition* ([arXiv:2211.09310](https://arxiv.org/abs/2211.09310)), on the ASD-ContextPose clips, with feature engineering applied to the model's inputs before any inference. Nothing here is implemented yet.

## The architecture, and what changes on our data

Deng et al. fine-tune a Kinetics-400-pretrained Video Swin Transformer (VST) on 30-frame clips cropped to the child by YOLOv5. During training, a frozen CLIP text encoder adds a cosine-similarity loss that pulls each video's feature towards a fixed text description of its behaviour class (loss = cross-entropy + λ · contrastive). The text branch is dropped at inference.

On our data, three things differ:

- **The target.** The paper classifies the behaviour (arm flapping, head banging, spinning, hand action). Our clips come from the same four categories, so the behaviour is already known. We predict the child's valence and arousal around an event instead: binary calls and regression of the consensus labels. The classification head becomes a valence/arousal head.
- **Time.** A 30-frame clip is about 1.2 s at 25 fps. Our labels describe a reaction across three windows: 5 s before the event (`FLANK_S` in `src/asd_contextpose/prepare.py`), the event itself (5 to over 30 s) and 5 s after.
- **Data size.** We have 153 development clips from 68 videos, against the paper's 1,437 clips from ESBD. Training a VST from scratch is not viable. Use the Kinetics-pretrained backbone, keep most of it frozen (fine-tune the last stage or use LoRA), and expect most of the gain to come from the inputs.

## Recommended input feature engineering

In order of expected value.

### 1. Event-aligned temporal sampling

Split each clip into the pre-event, event and post-event windows. Sample 8–16 evenly spaced frames from each window (TSN-style segment sampling), then either run three VST clips or one 32-frame clip that marks the window boundaries. Draw a different sample on each training epoch as augmentation, and average the predictions of several samples at test time.

Why: the label is about the reaction to the event. Consecutive 30-frame clips cover only a fraction of the event and none of the before/after contrast.

### 2. Child-centred crops from our pose tracks

Replace YOLOv5 with the child's ViTPose keypoints (`dataset/ViTPoseFilled`, or `vitpose_133` for whole-body). Build the box from the keypoints plus a margin, smooth its centre and scale over time, keep a fixed aspect ratio, and hold the last box when tracking drops.

Keep a second, low-resolution full-frame stream for the scene.

Why: YOLOv5 crops whichever person it detects, and social-interaction clips contain adults. The scene matters too: in the binary stream ablation (`tab:streams`), the best arousal callers are the context streams (VLM 76% and annotation 68% balanced accuracy).

### 3. High-resolution face crop

Add a face stream cropped at full resolution, gated by the existing face-quality checks:

- frontal (nose and both eyes tracked);
- YuNet detector score ≥ 0.8;
- brightness ≥ 40;
- sharpness (Laplacian variance) ≥ 10.

Mask the frames that fail the gate.

Why: the face is the only stream that consistently helps valence in our results. In a 224 px full frame the child's face is only a few pixels across.

### 4. Camera-motion compensation

Estimate the background's frame-to-frame motion with the child masked out (ORB features with a RANSAC homography) and warp it away before cropping.

Why: the videos are handheld home recordings. Without this, a camera pan looks like the child moving, which confounds spinning and any movement-based arousal cue.

### 5. Motion input

Add optical flow (RAFT) computed on the stabilised crops as a second VST stream, encoded as three channels (u, v, magnitude). Frame differences or motion-history images are cheaper alternatives.

Why: movement intensity carries arousal information. A real-time speed channel was the first input through which our networks used pose at all, lifting arousal CCC from about 0 to 0.17 (`tab:speed`).

### 6. Audio

Add log-mel spectrograms, or frozen audio embeddings, in a parallel branch.

Why: 86% of our clips have an annotated auditory event, and the paper's own discussion names sound as the next modality.

### 7. Language branch: per-clip or affect text

The paper's text is one fixed description per behaviour class. On our data that would encode the behaviour, which we already know, and say nothing about the individual clip. Better options:

- **Per-clip text, used only as a training target.** Use the event sentences (`dataset/ASD_ContextPose_Main_Event_Text_Natural.csv`, already set up for CLIP ViT-L/14) or the VLM scene descriptions. Because the text branch is dropped at inference, test clips need no annotation.
  - The event sentences are generated from the four event indicators, so they carry the same information as the annotation stream.
  - CLIP truncates text at 77 tokens, so long VLM descriptions need shortening.
- **Class-level affect prompts.** One description per valence/arousal quadrant, averaged over several phrasings (prompt ensembles).

### 8. Clip-level engineered features for the head

Cheap features that suit small data, concatenated before the prediction head:

- **Change features:** post-window minus pre-window movement energy, which directly measures a reaction.
- **Periodicity:** dominant frequency (FFT) and autocorrelation of wrist and head trajectories, which describe the repetitive behaviour's rate and regularity.

### 9. Appearance normalisation

- Resample every clip to a fixed frame rate.
- Apply CLAHE or gamma correction to dark clips.
- Normalise with the backbone's ImageNet mean and standard deviation. The paper scales frames to [0, 1], which does not match the pretrained weights.

### Low priority: pose heatmap volumes

Rendering skeletons as heatmap frames (as in PoseC3D) is background-invariant and works with 3D backbones, but pose alone added nothing in our stream ablations (`tab:streams`, `tab:streams_reg`). Try it only as an auxiliary input.

## Evaluation rules

- Fit anything learned (normalisation statistics, PCA, decision thresholds) on the training folds only.
- Fix the quality gates and crop parameters without looking at labels.
- Keep the video-grouped 5-fold cross-validation with duplicate videos merged (`DUPLICATE_VIDEOS` in `prepare.py`), and the 5-seed ensembles, so results compare directly with `tab:streams` and `tab:streams_reg`.
- To claim the model needs no annotation, use the event text only as a training-time target, never as an inference input.
- Leave the held-out test clips untouched until the configuration is fixed.

## Suggested order of work

1. Build the inputs for steps 1–3 (event-aligned sampling, pose-tracked child crops, gated face crops) and save VST-ready tensors per clip.
2. Train with per-clip text in the language branch (step 7) and compare with the same model without the text loss.
3. Add steps 4–6 one at a time, so each one's effect can be measured on its own.
