# Progress-update video script Austria or Australia?

**Target: ≤ 5:00 · Measured: 640 words ≈ 4:20 at a brisk 150 wpm, ≈ 4:55 at a slow 130 wpm.
Without the optional GenAI beat (35 words): ≈ 4:10–4:40. If a rehearsal runs over, drop
Beat 11 first.**
Everything below is backed by `Pair 1/AI4Vis/`, `Pair 2/country211-clip.ipynb`,
`ai_reflection_midpoint.md` and `ai-interactions/2026-11-07_review-of-prediction-explorer.md`.

| Beat | Slides | Time | ≈ Spoken | Words |
|---|---|---|---|---|
| Hook + setup | 1–2 | 0:00–0:35 | 35 s | 63 w |
| Who did what | 3 | 0:35–0:50 | 15 s | 53 w |
| Pair 1 method | 4 | 0:50–1:20 | 30 s | 76 w |
| Pair 1 findings | 5 | 1:20–2:00 | 40 s | 61 w |
| Pair 2 method | 6 | 2:00–2:35 | 35 s | 49 w |
| Pair 2 findings | 7 | 2:35–3:15 | 40 s | 98 w |
| Visual 1 (scatter) | 8 | 3:15–3:35 | 20 s | 43 w |
| Visual 2 (map) | 9 | 3:35–3:55 | 20 s | 32 w |
| Visual 3 (montage) | 10 | 3:55–4:15 | 20 s | 52 w |
| Synthesis | 11 | 4:15–4:30 | 15 s | 38 w |
| GenAI reflection *(opti l)* | 12 | 4: 4:35 | 10 s | 35 w |
| Conclusion + next steps | 13 | 4:35–4:50 | 15 s | 40 w |
| Safety buffer to 5:00 | — | 4:50–5:00 | — |

---

## Beat 1 · Slides 1–2 · 0:00–0:35
 
> Austria and Australia. One letter apart, half a world apart. Can a pretrained
> vision-language model tell them apart from a photo — or does it only know the postcards?
> Our model is CLIP ViT-B/32, used zero-shot. Our data is Country211: real, geotagged photos
> of both countries. And we split the work in two: one pair attacked the text side, one the
> image side.

**Visual:** title slide with the two dataset photos (alpine Austria / Australian beach),
then the problem slide.

## Beat 2 · Slide 3 · 0:35–0:50  
  
> We split the work in two. Pair one — Cristian and Alexander — took the text side: do the
> words in the prompt change the verdict? Pair two — Shamekh and Eiman — took the image side,
> without GenAI: where in the picture space do the errors actually live? Same model, two angles.

**Visual:** the two-column slide. **Needs confirmation:** confirm the two speakers and who
voices which pair before recording.

## Beat 3 · Slide 4 · 0:50–1:20  

> Pair one scored the same five hundred photos — two hundred fifty per country — under three
> label sets: Austria and Australia in English, Österreich and Australien in German, and a
> mixed pair. The Prediction Explorer shows each photo with one bar chart per label set, plus
> the top five of all two hundred eleven countries. Language Shift draws one dot per photo:
> the Austria score in English on the x-axis, in German on the y.

**Visual:** label-set chips + method bullets. If you have time, flash one real screenshot of
the Prediction Explorer window here.

## Beat 4 · Slide 5 · 1:20–2:00

> The finding: the label language moves the rdict. Accuracy drops from 86.6 percent in
> English to 76.8 percent in German. 482 of the 500 photos get a lower Austria score under
> German names, and 93 photos flip outright — English says Austria, German says Australia.
> 71 of those 93 are Austrian photos. So CLIP isn't reading “Austria”. It's reading the token.

**Numbers note:** 482/500 and 93 flips are printed in the committed scatter figure.
86.6 % / 76.8 % / 82.8 % (mixed) are derived from `cache_at_au/predictions.json` —
deterministic, but re-run `Language shift.py` once before recording if you want them printed.

## Beat 5 · Slide 6 · 2:00–2:35 

> Pair two embedded all six hundred photos — every split, nothing trained — and measured a
> 17 percent zero-shot error. Then they asked a different question: not how often, but where
> and why? A nearest-neighbour vote as image-side check, CLIP scene probes, k-means into nine
> clusters, a t-SNE map.

## Beat 6 · Slide 7 · 2:35–3:15 
 
> The pattern is sharp. Four clusters — Vienna streets, Australian coast, alpine snow, field
> hockey — are almost single-country and stay at or below seven percent error: 220 photos.
> The other four, 314 photos or 52 percent of the data, hold about 80 percent of the errors:
> night skies, conferences, graffiti, concerts. The errors aren't random: snow and buildings
> pull towards Austria, wildlife and night scenes towards Australia — 20 of the 27 errors
> shown are Austrian photos called Australian. And in the worst cluster, concerts are a
> text-side failure: the image neighbours do better than the prompt.

## Beat 7 · Slide 8 · 3:15–3:35
 
> Two figures carry the story. First, Language Shift. One dot per photo. Dots on the diagonal
> are unchanged by language — and the whole cloud sits below it. The grey box is the damage:
> 93 photos where only the wording decided the country.

**Visual:** `assets/language_shift.png` full-bleed.

## Beat 8 · Slide 9 · 3:35–3:55 

> Second, the embedding map. The X's are the errors, and they don't spread evenly — they pile
> up where the photos carry no place information, while the country islands stay almost clean.

**Visual:** `assets/pair2_map.png` full-bleed.

## Beat 9 · Slide 10 · 3:55–4:15

> The montage makes it concrete. A owy Australian forest called Austria. A toucan in an
> Austrian zoo called Australia. And the worst cluster is really two problems: portraits at
> 19 percent error, concerts at 50 — while the image neighbours sit at 27. The prompt throws
> away signal the image already has.

**Visual:** `assets/pair2_montage_stereo.png` (clusters 3 + 7) and
`assets/pair2_subclusters.png` (cluster 8 split).

## Beat 10 · Slide 11 · 4:15–4:30
 
> Put together: CLIP separates Austria from Australia when the photo shows a place. When it
> doesn't, it doesn't guess at random — it falls back on scene stereotypes, and the answer
> becomes unstable when you reword the promp 

## Beat 11 · Slide 12 · 4:30–4:35 *(optional — drop first if over time)*
 
> We were the AI-aided pair: ChatGPT shaped the idea, Claude Code reviewed the explorer —
> eight fixes proposed, two suggestions rejected. Useful, fast — and, as our reflection says,
> a bit of a black box.

## Beat 12 · Slide 13 · 4:35–4:50

> Next: the train split as a  ond dataset, and a click-through from the scatter into the
> explorer. The reflection and the interaction record are already in the repo. The video link
> goes in with the submission — thanks for watching.

---

## Delivery notes

- **Voices:** two speakers recommended (one per pair). **Needs confirmation:** who speaks.
- Record at 1080p in a quiet room; keep the deck fullscreen (`F11`), cursor hidden.
- Total is under 5:00 **only if** the optional at stays ≤ 15 s or is dropped. Rehearse once
  with a timer; if over, cut Beat 11 and shorten Beat 6's method list.
- Don't say numbers you can't see on the slide — the deck prints every number you need.

