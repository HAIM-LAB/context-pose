# what is the context lookup
_Exported on 11/11/2026 at 15:21:31 GMT-6 from Claude via WayLog_


**User**

what is the context lookup

---


**Claude**

Failed to authenticate: this login reached its time limit and cannot be renewed. Run `claude auth login` to sign in again, or use `claude setup-token` and set CLAUDE_CODE_OAUTH_TOKEN for unattended jobs

---


**User**

what is the context lookup

---


**Claude**

The context lookup is a baseline with no learning in it. It groups the training clips by their event context and predicts each group's typical label.

**How it works** ([train.py:32-45](src/asd_contextpose/train.py#L32-L45)):

1. Each clip comes with an event description (`event_text`). That sentence is generated from four binary indicators: `social_interaction`, `auditory_event`, `activity_object_change` and `visual_media_event`. All four at zero means "no event". So each clip falls into one of a small number of indicator combinations.
2. On the training fold, the lookup groups clips by that combination and stores:
   - **For regression:** the median valence and arousal of each group. That is why Table 2 calls it "median per combination". The median is used because scoring is by MAE.
   - **For the binary calls:** the mean annotator vote share of each group (share voting positive valence, share voting high arousal). The decision threshold is picked on the training clips.
3. Groups with fewer than 3 clips fall back to the overall median or share.
4. On a test clip, it looks up the clip's combination and returns that stored value. It never sees pose, face, video or audio.

**Why it keeps appearing in your tables:**

- **It is the bar to beat.** In regression, none of the neural models in Tables 2 and 3 beat it (0.425 mean MAE). It was also the best predictor on the held-out test set in Table 4 (0.389).
- **It is the starting point for the residual models.** "Residual" models, run with `--residual-by event_text`, start from the lookup's value and learn only a correction on top. That includes V3, F3, S4–S6 and the VLM rows in Table 9. In votes mode the starting point is the log-odds of the group's mean share ([train.py:48](src/asd_contextpose/train.py#L48)). So "V3 (context lookup + face)" means the lookup's prediction plus a correction from the face stream.
- **"Gain vs lookup"** in Tables 3, 5 and 6 measures how much that correction helps. Most of those gains have bootstrap intervals that include zero.

For the binary calls, the lookup is weak on valence (53% balanced accuracy, about chance) and moderate on arousal (68%). That fits the README's summary: arousal is read mostly from the event context, and valence mostly from the face.

---


**User**

difference btn binary calls and regression

---


**Claude**

They're two ways of scoring the same task: predicting a child's valence and arousal from a clip. The difference is what counts as the right answer and how errors are measured.

**Regression (Tables 2–6)**
- **Target:** a continuous valence score and a continuous arousal score per clip, built from the annotators' labels.
- **Question:** how close is the predicted number to the label?
- **Metrics:**
  - **MAE** (mean absolute error), where lower is better.
  - **CCC** (concordance correlation), where higher is better. It rewards predictions that track the labels' ups and downs and their overall level. A constant prediction scores 0, which is why "Overall median" shows 0.00.
- **Objective:** the models train on squared error (MSE) or L1 against the score. The lookup uses the group median.

**Binary calls (Tables 1, 7–10)**
- **Target:** two yes/no questions per clip:
  - Valence: positive vs not
  - Arousal: high vs low
- **Question:** does the model make the same call as the annotators?
- **Metrics:**
  - **Accuracy.** This is misleading here because the classes are unbalanced: most clips are positive valence and low arousal. Always predicting the majority already gets 67% and 78%.
  - **Balanced accuracy**, the average of the hit rate on each class. Chance is 50%.
  - **AUC**, how well the model's scores rank yes-clips above no-clips, ignoring the threshold. Chance is 0.50.
- **Objective:** models can still be trained with regression (the "R" rows in Table 7) and then thresholded. Or they can use the **votes objective** (the "V" rows, including V3): train directly on the share of annotators voting positive or high, with cross-entropy.
- **Ceiling:** the "Annotators" row scores each annotator against the other two. It shows how far humans agree on the call, so about 79% is a realistic upper bound.

**Why both are reported**

| | Regression | Binary calls |
|---|---|---|
| Measures | Exact level | Right side of the line |
| Lookup | Best overall, nothing beats it | Beaten by V3 and VLM models |
| Human ceiling | Not reported | ~77–81% |

The regression results are the negative finding: networks don't improve on the context lookup's numbers. The binary calls are where models show real signal beyond the lookup. V3 reaches 65% balanced accuracy on valence against the lookup's 53%. The VLM model reaches 0.77 AUC on arousal against the lookup's 0.63. The binary framing is also closer to how the labels were made (annotator votes), so it is less sensitive to noise in the exact score.

---


**User**

how are annotator accuracy calculated

---


**Claude**

Each annotator is scored as if they were a model, using the other two annotators as the answer key. The code is `annotator_ceiling` in [votes.py](src/asd_contextpose/votes.py).

**The setup.** Each clip was rated by three blind annotators.
- **Valence votes:** positive, neutral or negative. For the binary call, "not positive" means neutral or negative.
- **Arousal votes:** high or low.
- **"Unknown" votes** are treated as missing.

**The procedure, for each clip and each annotator *i*:**

1. Skip the clip if annotator *i*'s vote is missing, or if either of the other two votes is missing.
2. Turn the other two votes into binary calls, for example positive or not.
3. **If those two disagree, skip the clip** for annotator *i*, since there is no clear reference.
4. If they agree, their shared call is the label, and annotator *i*'s own binary call is the prediction.

All (clip, annotator) pairs that survive are pooled into one list. Accuracy and balanced accuracy are computed on that list, the same way as for the models: balanced accuracy is the mean of the hit rate on the "yes" class and on the "no" class.

**Example (valence):**

| Clip | A | B | C | Pairs scored |
|---|---|---|---|---|
| 1 | pos | pos | neutral | A vs pos ✓, B vs pos ✓; C skipped (A and B... wait, C is scored: A and B agree on pos, C said not → ✗) |

Here is that example worked out properly:

- **Clip 1: A = positive, B = positive, C = neutral**
  - A is checked against B and C, who disagree, so A is skipped.
  - B is checked against A and C, who disagree, so B is skipped.
  - C is checked against A and B, who agree on positive. C said "not positive", so that pair is a miss.
- **Clip 2: all three positive.** All three pairs are scored, and all three are hits.

**Why AUC is "–".** An annotator gives a hard yes/no vote, not a score, so there is nothing to rank and no AUC to compute.

**Caveat when comparing to the models.** The two references are not quite the same:
- **Models** are scored against the majority of all three annotators. Only ties are dropped, which happens on valence when there is a missing vote.
- **Annotators** are scored only where the other two agree. That drops exactly the ambiguous clips where a disagreement is most likely.

So the annotator figures of 79% and 77% are measured on an easier subset. Read them as a rough ceiling, not a like-for-like comparison. The true agreement on the clips the models are scored on is probably a bit lower.

---


**User**

where is this coming from:
Audio, AST embedding and vjepa 2

---


**Claude**

The report mentions them, but the repo has no extraction code. I'll check the session logs and data folders.

---


**Claude**

The log shows two extraction scripts were written. I'll find out where they were saved.

---


**Claude**

Both rows of Table 5 come from a one-off experiment in an earlier Claude Code session on your Windows machine. The code for it was never added to this repo.

**Where the code lived.** That session wrote two scripts in its temporary scratchpad folder, outside the repo:
- `C:\Users\hfansah\AppData\Local\Temp\claude\c--Users-hfansah-workspace-cvpr\19e02c99-...\scratchpad\media_features.py` extracted the frozen embeddings.
- `...\scratchpad\media_probe.py` fitted the linear correction on the lookup and wrote Table 5.

According to the session log, the extracted features and results went to `checkpoints/media/`, which is gitignored. Nothing in `src/` produces these features. The only places they appear here are the report ([asd_contextpose_report.tex:442-460](reports/asd_contextpose_report.tex#L442-L460)) and the session log ([.waylog/history/...pytorch-gpu.md:2735-2890](.waylog/history/2026-11-03_18-46-22Z-claude-setup-the-project-with-pytorch-gpu.md#L2735)).

**What the session did:**
- **V-JEPA 2:** a pretrained self-supervised video encoder from Meta, run frozen on a crop around the child. The crop was centred using the pose tracks, and the preprocessing was done by hand because torchvision wasn't installed. It ran on CPU at about 2.3 s per window.
- **AST (Audio Spectrogram Transformer):** a pretrained audio model, run frozen on each clip's soundtrack (AAC, 48 kHz). It gave two inputs:
  - "AST embedding": the model's internal feature vector.
  - "AudioSet probabilities + loudness": its scores for the AudioSet sound classes (crying, laughter and so on), plus a loudness measure.
- **The probe:** each embedding was used as an L1 linear correction on the per-combination median lookup. It used the same participant-grouped 5 folds as the rest, and only the 153 development clips.

**What I couldn't confirm.** The log never names the exact checkpoints (for example which V-JEPA 2 size, or which AST fine-tune). They would only be in `media_features.py`. The report also doesn't cite either model.

**Practical consequence.** If the report or paper includes Table 5, these rows can't be reproduced from this repo. To fix that:
1. Copy `media_features.py` and `media_probe.py` from the Windows scratchpad into the repo, if that temp folder still exists.
2. Record the exact checkpoint IDs.
3. Add `av` (PyAV) as a dependency, since the extractor needs it.

Otherwise, label the rows as an exploratory probe. Both inputs showed no gain, so dropping them wouldn't change any conclusion.

---


**User**

what is the numbers in the square braces

---


**User**

[Image: source: /private/tmp/claude-501/-Users-henryansah-workspace-cvpr/57a11c9b-46ac-49c8-93c6-029258152456/images/1.png]

---


**Claude**

The numbers in square brackets are **95% confidence intervals**. They show how much each number could move if the study were repeated with a different set of children.

So for the VLM description embedding on arousal:
- **72% [63, 85]:** the balanced accuracy is 72%, and the plausible range is about 63% to 85%.
- **0.77 [0.66, 0.87]:** the AUC is 0.77, and the plausible range is about 0.66 to 0.87.

**How they're computed** (`bootstrap_interval` in [votes.py](src/asd_contextpose/votes.py)). This is a participant bootstrap:

1. Take the cross-validated predictions for all clips, grouped by participant.
2. Draw a new set of participants at random **with replacement**, the same size as the original. Some participants appear twice and some not at all, and each drawn participant brings all of their clips.
3. Recompute balanced accuracy or AUC on that resampled set.
4. Repeat 2,000 times.
5. The bracket is the 2.5th and 97.5th percentiles of those 2,000 values.

Participants are resampled instead of single clips because clips from the same child are correlated. Resampling clips would make the intervals look falsely narrow. One caveat: the data has no real participant IDs, so the code uses the **video ID** as "participant" ([README](README.md) notes this). The same child can appear in more than one video, so the intervals may still be slightly too narrow.

**How to read them:**
- **They're wide.** A range of 63–85% reflects only 153 clips, about 30 of them high arousal. Treat every result in these tables as rough.
- **They overlap.** V3's arousal interval [57, 79] overlaps the VLM's [63, 85]. So 72% vs 69% isn't evidence that the VLM model is better than V3.
- **Comparison against chance is meaningful.** The VLM interval sits entirely above 50% and 0.50, so the model clearly beats guessing.
- **Comparison against the lookup needs a different interval.** To show a model beats the lookup, you want an interval on the *difference* between them. The code computes this as `model_vs_lookup_balanced_accuracy_95ci`, but Table 1 doesn't report it. Tables 3, 5 and 6 do: that's the bracket in their "Gain vs lookup" column. Most of those gain intervals include zero.

---


**User**

which input streams go into the V3 model

---


**Claude**

The README gives V3's settings. I'll check in the code whether the context indicators also go into the network with `--inputs face`, or only through the lookup.

---


**Claude**

V3 uses **two inputs: the event context and the face**. They enter the model in different ways, and pose is not used at all.

**1. Event context, through the lookup only**
- The flag `--residual-by event_text` builds the lookup table on the training clips. For each indicator combination it stores the log-odds of the mean annotator vote share (`group_logits`, [train.py:48](src/asd_contextpose/train.py#L48)).
- That value is added to the network's output as a fixed offset, the starting point.
- The four context indicators are **not** an input to the network itself. With `--inputs face`, the forward pass returns before the context projection is used ([model.py:211-212](src/asd_contextpose/model.py#L211-L212)).

**2. Face stream, the only part that learns**
- **Features:** frozen FER ViT embeddings (`trpakov/vit-face-expression`, 768-dimensional [CLS]). There is one per step, 32 steps in each of the pre-event, event and post-event windows, plus a 0/1 quality-gate mask.
- **Gate:** a step is kept only if the face is frontal (nose and both eyes tracked), the YuNet detector score is at least 0.8, and the crop is not dark or blurred.
- **Network** (`_face`, [model.py:239-256](src/asd_contextpose/model.py#L239-L256)):
  1. Each step is encoded with `LayerNorm → Linear(768, 32) → GELU`.
  2. The kept steps are averaged within each of the three windows.
  3. The three averages are concatenated, and a small head maps them to a correction for **both** valence and arousal (`--face-targets both`).
- **Training details:**
  - The correction is zero for clips with no usable face, so those clips just get the lookup value.
  - `--face-dropout 0.3` hides a training clip's face 30% of the time.

**So V3's prediction is:**

```
log-odds of positive/high = lookup(event combination) + face correction
```

It is trained with the votes objective (cross-entropy against the annotator vote shares). The final model is an ensemble of 5 seeds.

**Pose** is still loaded and must be present in the manifest: the pose flags `--num-joints 11 --joint-features 4` and the `--speed` prepare step are in the recipe. But the face-only forward pass never reads it. The README states it plainly: "pose did not help".

That matches the README's interpretation: valence comes mostly from the face, arousal mostly from the context lookup.

---


**User**

EmotionCLIP. how is it  getting results for this?

---


**Claude**

Table 8 also came from the earlier Windows session. I'll pull the EmotionCLIP details from its log.

---


**Claude**

Table 8 also comes from the earlier Windows session. Like Table 5, its extraction and probe code is not in this repo. Here is what the session log and the report ([asd_contextpose_report.tex:574-605](reports/asd_contextpose_report.tex#L574)) describe.

**1. The model, used frozen with no fine-tuning**
- **What it is:** EmotionCLIP (CVPR 2023, github.com/xeaver/emotionclip). It is OpenCLIP's ViT-B/32 image encoder plus a 6-layer temporal transformer, trained on YouTube vlogs and their transcripts to read emotion from a person's body, face and scene.
- **Setup:**
  - The repo was cloned to `third_party/EmotionCLIP` at commit `ca142dd`.
  - The 2 GB weights were cached at `~/.cache/asd_contextpose/emotionclip/emotionclip_latest.pt`.
  - torchvision was stubbed so the model would load without changing your dependencies.
- **Fine-tuning was ruled out:** 170M parameters against about 150 clips would just memorise.

**2. Feature extraction** (`emotionclip_features.py`, in the Windows scratchpad)
- **Subject mask:** EmotionCLIP takes a mask marking the person to focus on. The session built it from your pose tracks: the child's box from the joints, padded by 0.3, with the torso extended. If the child wasn't tracked, the mask was left empty. An earlier fallback had put the "subject" on the TV.
- **Crop:** a 224 px square centred on the child. The child was tracked in 94% of steps.
- **Outputs per clip,** cached in `dataset/EmotionCLIPFeatures/`:
  - a 512-d frame embedding at each of the 32 steps in each of the pre, event and post windows;
  - a window-level embedding from 8 frames through the temporal transformer.

**3. "Linear" rows** (`emotionclip_probe.py`, `emotionclip_robust.py`)
1. The embeddings were reduced with **PCA to 8, 16 or 32 dimensions**.
2. A linear classifier was fitted, alone or alongside the context indicators and face features.
3. It was scored on the same participant-grouped, deduplicated folds, with thresholds chosen on the training folds.
4. The whole procedure was repeated over **5 different fold arrangements** (fold seeds).

That is why these rows show **ranges**: "74–75%" is the average across arrangements for 8–16 dimensions. The bracket "(65–79%)" is the spread across arrangements, **not** a bootstrap interval. At 32 dimensions it overfit, dropping to 68%.

**4. "Network" rows (called "E3" in the log)**
- This is V3's exact recipe with the EmotionCLIP per-step embeddings in the face slot instead of the FER embeddings: `--face-dim 512 --face-targets both --objective votes --residual-by event_text --inputs face`, with data in `data/vitpose_eclip`.
- It was run as a 5-seed ensemble, so these rows get proper bootstrap intervals, like V3.

**How much to trust it:**
- **Arousal signal looks real.** EmotionCLIP alone reaches AUC 0.72–0.77, against 0.63 for the lookup.
- **The 74–75% linear figure is optimistic.**
  - The PCA size was chosen after seeing results.
  - The network version of the same idea reached only 60% balanced accuracy, though with AUC 0.75.
  - There is no participant-bootstrap interval for the linear rows.
- **Valence is below chance** (AUC 0.35–0.48). That pattern means the features pick up cues tied to particular videos or children that reverse in held-out folds. Since EmotionCLIP sees the whole child and room, **identity or room leakage** across videos is a real concern. The data has no participant IDs to rule it out.
- **Not reproducible from the repo.** The extractor, probe scripts and loader exist only in that session's scratchpad. The weights also have no stated licence, which needs checking before publication.

---


**User**

explain the input to the emotionclip

---


**Claude**

EmotionCLIP takes two inputs for each frame: the **image** and a **subject mask** that says which person to read. The log notes these details were checked against the EmotionCLIP code (`src/models/base.py`) before extraction.

**1. Frames (RGB images)**
- **Size:** 224 × 224 px with ImageNet normalisation, the same as the EmotionCLIP training data.
- **Source:** each clip's video. 32 time steps are sampled in each of the three windows (pre-event, event, post-event), the same steps as the pose and face streams.
- **Crop:** a square crop centred on the child, with side 0.875 × the frame's short side. It keeps the child large while keeping some scene around them, which EmotionCLIP is meant to use. The crop is then resized to 224 px.

**2. Subject mask (binary, one per frame)**
- **What it does:** a frame often contains several people, such as the child, a parent or a person on a TV. The mask tells the model which one is the subject. Inside the model, the mask is average-pooled over each 32 × 32 patch of the image, and the result is added to that patch's position embedding. So patches covering the child are marked "this is the subject" while the whole image is still visible.
- **How it was built here** (EmotionCLIP was trained with YOLOv7 person boxes; your pose tracks were used instead):
  1. Take the child's tracked joints at that step. If the step has no joints, look up to ±2 frames away.
  2. Draw the joints' bounding box and pad it by 0.3 × body scale.
  3. Extend it downward by 2× the torso, since the 11 joints are mostly upper body.
  4. Fill that rectangle with 1s and the rest with 0s, in the same crop coordinates as the frame.
  5. **If the child isn't tracked, leave the mask empty (all zeros).** An earlier version fell back to the median box, and that sometimes put the subject on the TV.

**3. Two ways the model was run**

| Output | Input | Path through EmotionCLIP | Shape cached |
|---|---|---|---|
| Frame embedding | 1 frame + its mask | ViT-B/32 image encoder | `[3 windows, 32 steps, 512]` |
| Window (video) embedding | 8 frames + masks, evenly spaced across the window | Image encoder on each frame, then the 6-layer temporal transformer | `[3 windows, 512]` |

A `tracked` flag `[3, 32]` records which steps had the child. Embeddings were computed in fp16 and L2-normalised, and saved in `dataset/EmotionCLIPFeatures/`.

**Where each output was used:**
- **Linear probes:** pooled per-window embeddings, reduced with PCA.
- **Network (E3):** the per-step frame embeddings plus the tracked flag as a mask, put in V3's face slot (`[32, 513]` per window).

**What EmotionCLIP does *not* get:**
- **No text.** Its text encoder (trained on transcripts) wasn't used, so the event descriptions never went into it.
- **No audio.**
- **No pose numbers.** Pose only decides where the crop and mask go.

**Caveat:** the mask covers the child's whole body region and the crop keeps the room. So the embedding mixes body language, face, clothing and setting. That is probably why it helps arousal (energy and posture show up there) but fails on valence, and why identity or room leakage across videos is a concern.

---
