# explain whats in here
_Exported on 11/13/2026 at 16:33:08 GMT-6 from Claude via WayLog_


**User**

explain whats in here

---


**User**

[Image: source: /private/tmp/claude-501/-Users-henryansah-workspace-cvpr/8e57bbab-c304-46c0-98c5-27b26c744e65/images/1.png]

---


**Claude**

This is a 2×2 grid laid out like the valence–arousal plane. Valence runs left to right (not positive → positive) and arousal runs bottom to top (low → high). Each panel is one quadrant of the ground-truth labels. Each bar is **recall within that quadrant**: of the clips whose true label falls in that quadrant, the percentage a method also placed there. The dashed line marks 25%, which is chance for 4 quadrants.

## What's being compared

- **Hatched grey bars are references, not models.**
  - *Annotators (each vs other two)* is how often one human annotator agrees with the other two. It's roughly the ceiling you could expect.
  - *Annotation lookup* is a baseline with no model behind it.
- **Blue bars are the model variants**, built from combinations of pose, annotation, face and VLM (vision-language model) inputs.

## Main takeaways

1. **The classes are very unbalanced.** There are 145 clips: 72 positive/low, 40 not-positive/low, 28 positive/high and only 5 not-positive/high. In the top-left panel, one clip is worth 20 points. So the 0 / 20 / 60 values there mean 0, 1 or 3 clips correct and tell you almost nothing. In the top-right panel, one clip is worth about 3.6 points.

2. **Humans don't agree much either.** Annotator agreement is 75 / 57 / 75 / 45. Positive/low arousal is the hardest quadrant even for people (45%). They probably disagree on arousal more than on valence.

3. **No single model wins every quadrant. Each one leans toward certain quadrants.**
   - **Annotation + face (V3)** is best on positive/high (64), slightly above human agreement (57).
   - **Annotation + VLM** is best on positive/low (61), well above humans (45), but drops to 25 on positive/high. That pattern suggests it tends to call positive clips "low arousal". It scores well in the big class at the cost of the smaller one.
   - **Not-positive/low** is fairly stable across models at 50–62. All of them are still below human agreement (75).

4. **Adding pose doesn't help and seems to hurt.**
   - *Pose + annotation + face (full)* is no better than the variants without pose, and gets 0 on top-left.
   - *Pose only* is near chance everywhere except the n=5 cell. Its 60 there is 3 clips, which is noise.

5. **Averaging the four recalls gives a rough summary (balanced accuracy):**

| Method | Mean recall |
|---|---|
| Annotators | ~63 |
| Annotation + face (V3) | ~43 |
| Annotation + VLM + face | ~42 |
| VLM + face | ~40 |
| Annotation + VLM | ~39 |
| Annotation lookup | ~33 |
| Pose only | ~32 |
| Full (with pose) | ~28 |

   The face-based variants beat the lookup baseline. Everything is well short of how far humans agree with each other.

## Caveats when reading it

- **Recall leaves out false positives.** A model that put every clip in positive/low would score 100 in that panel. Read this chart alongside precision or a confusion matrix.
- **Small differences mean little here.** In the top two panels, a gap of a few points is only one or two clips. It would help to add confidence intervals or the raw counts (k/n) to the labels.

---


**User**

create graph again removing annotators  and lookup bars

---


**Claude**

The script that generated this figure isn't in the repo, so I'll rebuild the chart with matplotlib using the values from the table in `test.tex`.

---


**Claude**

The values match the table, so I'll write a script that makes a models-only version. It saves to a new file so the original figure stays untouched.

---


**Claude**

The project's virtualenv doesn't have matplotlib. I'll run the script in a temporary uv environment rather than install it into the venv.

---


**Claude**

I've remade the chart without the annotator and lookup bars. It now shows only the six model variants, in the same 2×2 quadrant layout, with the same values (taken from the quadrant table in `reports/test.tex`). The subtitle now just explains the chance line.

The new files are:
- [quadrant_recall_models.png](reports/figures/quadrant_recall_models.png) and [quadrant_recall_models.pdf](reports/figures/quadrant_recall_models.pdf)
- [quadrant_recall_models.py](reports/figures/quadrant_recall_models.py), the script that makes them

**About the script:** whatever made the original figure isn't in the repo, so I wrote a new one with the recall values copied in by hand. The project's `.venv` doesn't have matplotlib, so I ran it in a throwaway environment that doesn't touch the venv:
```
uv run --no-project --with matplotlib python reports/figures/quadrant_recall_models.py
```

**Not changed:** the original `quadrant_recall.pdf` is still there, and `test.tex` (Figure `fig:quadrant_recall`) still points to it and its caption still mentions hatched reference bars. If you want the paper to use the new figure, I can change the `\includegraphics` path and update the caption.

---
