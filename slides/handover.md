# Handover Phase I midpoint (The Visionaires)

Repo: <https://github.com/cristianknebel/AI-and-Visualization> · Deadline: **11 Nov 2026, 23:59** · Video: **≤ 5:00**

## 1. One-paragraph project summary

The project asks whether OpenAI's CLIP ViT-B/32 can tell Austria from Australia from images
alone, or whether it relies on stereotypical visual cues. CLIP is used strictly zero-shot on
Country211 photos of both countries. The team split the question in half: one pair (AI-aided)
studied the **text side** how the wording and language of the class labels change CLIP's
scores, using two matplotlib tools, a *Prediction Explorer* and a *Language Shift* scatter —
while the other pair (no LLMs) studied the **image side** embedding all photos, mapping the
space with t-SNE, clustering it with k-means, and locating where and why errors concentrate.
Both halves converge: CLIP separates the countries when the photo shows a place-specific scene,
and falls back on scene stereotypes otherwise.

## 2. Evidence-based contribution of each pair

**Pair 1 text side, AI-aided (Cristian Knebel K52104480, Alexander Strasser K01408428)**
Evidence: `Pair 1/AI4Vis/Prediction explorer.py`, `Language shift.py`, `cache_at_au/`
(500 photos: 250 AT + 250 AU, train 300 + test 200), `language_shift_scatter.png`.

- Built `Prediction explorer.py`: streams Country211 (`clip-benchmark/wds_country211`), caches
  AT/AU images + CLIP predictions, scores each photo under **3 label sets** (EN, DE, mixed)
  plus the top-5 of all 211 countries, with a Tk/matplotlib browser (✓/✗ verdict, close-call
  margin, "Only mistakes").
- Built `Language shift.py`: scatter of Austria score EN (x) vs DE (y) per photo; prints
  "482 of 500" below-diagonal and "93 photos" flipped (confirmed in the committed PNG).
- Ran the AI review cycle (see §GenAI).

**Pair 2 image side, no LLMs (Shamekh Al-Suwi K12146739, Eiman Mohamed K12249316)**
Evidence: `Pair 2/country211-clip.ipynb` (committed 2026-11-11, "No LLMs Pair NB").

- Embedded **600 photos** (150/50/100 per country across train/valid/test) into 512-d CLIP
  space; measured **0.17 zero-shot error** with "a photo taken in …" prompts.
- Mapped and clustered the space: t-SNE figure, k-means with k=9 chosen by silhouette (0.05),
  10-NN leave-one-out image-side check, 14 CLIP scene probes (labelled unreliable in-notebook),
  per-cluster summary table + montages, and a sub-cluster analysis of the worst cluster.

**Needs confirmation:** Pair 2's README lists only student IDs, no names; the ID→name mapping
comes from `team-info.json`. Confirm before naming anyone on camera.

## 3. Confirmed findings

Pair 1 (from committed figure + cached predictions):

- **482 / 500** photos get a lower Austria score with German names; **93 photos** flip
  (EN Austria → DE Australia); printed in `language_shift_scatter.png`.
- Accuracy **EN 86.6 %** (433/500) → **DE 76.8 %** (384/500); mixed 82.8 % (414/500);
  **139/500** wrong under ≥ 1 label set; flips split 71 AT / 22 AU.
  *Derived from `predictions.json` deterministic; re-run `Language shift.py` before quoting
  on camera if you want them printed by the code itself.*

Pair 2 (from the notebook's printed outputs and interpretation cell):

- Zero-shot error **0.17** (photo template) → 102 of 600 photos wrong; split **69 AT vs 33 AU**
  (cell 28 output; consistent with 17 %).
- Clusters 0/2/3/6 = **220 images, error ≤ 7 %**; clusters 1/4/5/8 = **314 images (52 %) holding
  ≈ 80 % of errors**.
- Sparser clusters err more: density 0.75 → 0–2 % error; 0.633/0.641 → 35 % / 24 %.
- Stereotype asymmetry: snow/buildings → Austria; nature/wildlife/night → Australia;
  **20 of 27 shown errors** are Austrian photos predicted Australian.
- Cluster 8 (52 images) splits in two: portraits **19 %** error vs concerts **50 %** (always-
  Austria baseline 35 %, neighbours 27 %) → **text-side failure**; 35–46 % of images in
  clusters 4/5/8 flip when only the prompt wording changes.

## 4. Missing or contradictory information (**needs confirmation**)

1. **Video link** not yet produced; must be uploaded (unlisted YouTube / Drive / uni storage).
2. **Speakers** who voices which pair is not recorded anywhere.
3. **Pair 1 accuracies** (86.6 / 76.8 / 82.8 %) are derived from the cache, not printed in any
   repo file re-run `Language shift.py` or accept the derivation.
4. **README says "300 per country"** for Country211; Pair 1 used 500 (train+test) and Pair 2
   used 600 (all splits). Not contradictory (different slices), but say it explicitly on video.
5. **`Pair 1/README.md` contains `: K1408428`** a typo of Alexander's ID (`team-info.json`
   has the correct `k01408428`). The interaction record documents this; the repo file is unfixed.
6. **`Pair 2/README.md` has no names**, only `K12249316` and `K12146739`.
7. **No `requirements.txt`** in the repo (proposed by the AI review, still open).
8. **`ai-interactions/` holds one file** with two entries (Claude Code review, 2026-11-07;
   ChatGPT planning, 2026-11-11). Pair 2 has no interaction record consistent with its
   "no LLMs" design, but state that explicitly.
9. **The reflection covers accepted suggestions**; rejected ones and verification habits come
   only from the interaction record. Cite both files on the reflection slide.
10. **Pair 1's photo count** commit message says "All images taken (500, the 100 up to 600
    are for validation)": the validation split exists in the script's `EXPECTED` map but was
    never cached. Don't claim "600 photos" for Pair 1.

## 5. Recommended visuals for the video

| Time | Visual | Source |
|---|---|---|
| 0:00–0:35 | Two dataset photos side by side (alpine AT / beach AU) | `cache_at_au/images/Austria_train_007.jpg`, `Australia_test_003.jpg` |
| 0:50–1:20 | Label-set chips + Prediction Explorer UI description (or a live screenshot) | `Prediction explorer.py` |
| 3:15–3:35 | Language Shift scatter (full-bleed) | `Pair 1/AI4Vis/language_shift_scatter.png` |
| 3:35–3:55 | t-SNE map + k-means clusters (full-bleed) | notebook Figure 1 |
| 3:55–4:15 | Montage rows for clusters 3 & 7 + cluster-8 sub-cluster strip | notebook Figures 3–4 |
| 4:35–4:50 | Submission checklist card | this handover |

All of these are pre-extracted in `slides/assets/` and embedded in `slides/index.html`.

## 6. Video script with timestamps

See **`slides/script.md`** full spoken script, 12 beats, 0:00–4:50 + safety buffer to 5:00,
with per-beat visuals and word counts. Key timings:

- 0:00–0:35 problem · 0:35–0:50 split of work
- 0:50–2:00 Pair 1 (method 0:50–1:20, findings 1:20–2:00)
- 2:00–3:15 Pair 2 (method 2:00–2:35, findings 2:35–3:15)
- 3:15–4:15 the three key visuals
- 4:15–4:30 synthesis · 4:30–4:35 optional GenAI beat · 4:35–4:50 conclusion

## 7. Recording and submission checklist

See **`slides/recording-checklist.md`** for the full version. Short form:

- [ ] Confirm speakers + optional beat cut/kept
- [ ] Re-run `Language shift.py` once (fresh PNG + printed stats)
- [ ] Record ≤ 5:00 (aim 4:30–4:50), check the file duration
- [ ] Upload unlisted, verify the link in a private/incognito window
- [ ] Repo contains: `team-info.json`, `workload.json`, `README.md`,
      `ai_reflection_midpoint.md`, `ai-interactions/`, pair folders all ✅ today
- [ ] Paste repo link + video link into the submission form before 11 Nov, 23:59

## GenAI reflection summary (for the optional beat / report cross-check)

- **Tools:** ChatGPT (GPT-6) for the project idea and visualisation choice; Claude Code
  (Claude Opus 5.5 in the terminal, plus two extra review agents) for the code review.
- **Influence:** ChatGPT fixed model (`openai/clip-vit-base-patch32`), dataset
  (`clip-benchmark/wds_country211`) and both visualisations (Language Shift, Prediction
  Explorer). Claude Code proposed 8 robustness fixes (cache path, offline class
  names, Tk crash, percent-label placement, grid rows, button references, stale predictions…).
- **Rejected:** uv project setup ("use his setup"), dropping `.eval()`, and a measured-and-
  rejected re-encoding of 217 prompts (7 s vs 2 s, runs once). Still open: JPEG re-encoding,
  train-split extension.
- **Verification:** independent agents reviewed code and dataset separately; Alexander steered
  with prompts like "don't overwrite his work, keep it simple, verify everything"; a fixes
  branch (`review/robustness`) was pushed while the record stayed transparent.
- **Learning vs black box (quoted from the reflection):** "it was not really a learning
  experience … it felt like a black box but it was easy to get the results"; "AI gives you the
  illusion of finishing the work"; generating ideas is easy, choosing the fitting ones is hard.

