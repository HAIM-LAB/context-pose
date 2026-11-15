# write a research paper methodology just talking ab
_Exported on 11/11/2026 at 17:29:57 GMT-6 from Claude via WayLog_


**User**

write a research paper methodology just talking about the architecture and its components (context-pose) in latex. add the equations as well

---


**Claude**

I'll start by looking at the existing reports and the ContextPose model code so the write-up matches the real architecture.

---


**Claude**

The code and the existing Markdown write-up give me the full architecture. Next I'll check the report's LaTeX preamble so the new file matches it.

---


**Claude**

The `which` call failed only because no LaTeX binary is on the Bash path. I'll check for MiKTeX on Windows and get exact parameter counts from the model.

---


**Claude**

I have every detail I need, so I'll write the LaTeX file now.

---


**Claude**

I'll fix a few details: `\mathbb{1}` renders badly with amssymb, one figure label overlaps an arrow, and two sentences need tightening.

---


**Claude**

Written for: readers of a CVPR-style paper (reviewers and other researchers), not for people working in this codebase.

I've written the ContextPose methodology section as a standalone LaTeX file: [reports/contextpose_methodology.tex](reports/contextpose_methodology.tex). I haven't compiled it, because there is no LaTeX install on this machine. Run it through `pdflatex` (or Overleaf) once before using it.

Every equation was checked against [model.py](src/asd_contextpose/model.py), the offset code in [train.py](src/asd_contextpose/train.py) and the speed channel in [prepare.py](src/asd_contextpose/prepare.py). It uses the same preamble and diagram styles as the existing report, so you can paste the `\section` into the main paper. The section covers:

- **Inputs:** how poses are normalised and resampled to 32 steps per window, plus the speed channel and its cap.
- **Pose encoder:** the two convolution layers, shared across the three windows.
- **Context encoder:** the three context forms (the four indicators, CLIP, the VLM embedding), the linear projection, and the learned region embeddings.
- **Context-queried attention:** the per-head attention equations, and why the context picks which time steps are summarised.
- **FiLM variant:** the alternative where the context scales and shifts the pose features instead.
- **Region Transformer and head:** one Transformer layer over the three windows, averaging, the optional context skip, and the MLP head.
- **Face stream:** encoding, averaging over quality-gated steps, and the three safeguards (zero initialisation, zero output when no face passes the gate, face dropout).
- **Residual offset:** the output is a correction to a lookup by event-indicator combination, with groups under three clips falling back to the overall value.
- **Objective:** cross-entropy against the annotator vote shares, with regression as an alternative.
- **Stream ablations:** the pose-only, context-only and face-only forms.
- **Hyper-parameter table:** includes parameter counts measured from the code: 73.6k without the face stream and 102.9k with it.

The section also includes a TikZ architecture diagram.

The paper text has to describe the model as the code builds it, so two choices in the code are stated plainly in the section:
- **No positional encoding.** Time order inside a window comes only from the convolutions, which each see a few neighbouring steps.
- **Post-norm Transformer with ReLU.** This is PyTorch's default, while the other layers use GELU.

The section sticks to architecture, as you asked. It leaves out the data pipeline, training settings, the GRU/BiLSTM baselines and evaluation; those are in [methodology.md](reports/methodology.md) if you want them added.

---
