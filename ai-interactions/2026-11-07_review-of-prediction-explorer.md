# 2026-11-07: Claude Code reviews the prediction explorer

- Who: Alexander Strasser (AI-aided pair, Phase I)
- Tool: Claude Code with Claude Opus 5.5 in the terminal. Two extra Claude agents were started for independent checks: one reviewed the code, the other checked the dataset.
- Input: our repo at commit `28e23fe` (Cristi's prediction explorer from 4 Nov), the assignment text, our team chat

## Important prompts

Main prompt (shortened, wording kept):

> "I want you to burn all the tokens from this session to verify everything is done: Push it, pull it, and set it up in an IDE. Set up a test suite. Inspect every code line and every idea. Find alternatives, counterfactuals, or variations. Reverify them so I can give him options. Stay probably with the idea of what he implemented, but just find the simplest stuff, maybe simplifications or readability improvements, and try to make it robust."

Follow-up, after the AI had set up its own environment with `uv`: "use his setup". The AI then used plain Python 3.14 + pip + PyCharm, like Cristi.

## Suggestions and what happened to them

| Suggestion by the AI | Decision | Why |
|---|---|---|
| Set up the project with uv | rejected by Alexander: "use his setup" | the AI then set it up like Cristi (pip + PyCharm) and added no uv files to the repo |
| Fix: cache path relative to the script, not the current folder | proposed, Cristi decides | started from the repo root, it re-downloaded the dataset |
| Fix: fetch class names only when needed | proposed | without internet the window did not open, although all data is in the repo |
| Fix: catch the error when Tk can't maximize | proposed | crash on Linux |
| Fix: percent labels always just after the bar | proposed | 109 of 1200 labels were drawn across the bar end |
| Fix: wider gap + shorter long country names | proposed | names ran into the photo when the window is not maximized |
| Fix: grid rows from the number of label sets | proposed | a 4th label set overlapped the top-5 chart |
| Fix: keep references to the buttons | proposed | buttons died in IPython/Jupyter |
| Fix: recompute predictions that don't match the photos | proposed | stale predictions could be shown next to the wrong photo |
| Add README.md and requirements.txt | README proposed, then removed on the fixes branch because Eiman's README is the team's; requirements.txt proposed | the course needs a README; Eiman's covers task, model and dataset |
| Encode the 217 prompts once instead of per batch | rejected by the AI after measuring | 7 s vs 2 s, and it runs only once before the cache takes over |
| Optional: shade the parts of the photo CLIP never sees | proposed, on the separate branch `review/optional` | CLIP keeps only the centre square, so the median photo loses 33% of its pixels. The agent's idea was to show only the square; the AI chose shading to keep the context |
| Optional: `logits_per_image` and plain-Python argmax, no numpy | proposed, on the separate branch `review/optional` | same results to 1e-5, fewer lines |
| Agent's suggestion: `wspace=0.5` against the overlapping names | modified | `0.35` plus shortening names over 22 characters keeps the photo bigger, with 0 overlaps |
| Agent's suggestion: drop `.eval()` | rejected | already the default, but the line makes it explicit |
| Save the original JPEG bytes (no re-encoding) | open, for the pair | faithful to the dataset, but the cache grows to 35 MB and the predictions change |
| Use the train split as well (500 photos) | open, for the pair | patterns are confirmed on it, needs about 300 MB more download |

## Steering prompts and what they changed

Alexander's prompts, quoted as typed. Each line says what changed.

- "K01408428 k1408428": the student ID was checked. The team-info.json on GitHub (uploaded by the other pair on 2026-11-08) still has k1408428. Not corrected yet: the fix is in team-info.json, which this commit leaves alone.
- "don't overwrite his work, keep it simple, verify everything. you sure": the AI had written "keep it simple" as if it were Alexander's brief, and gave a reason for the uv decision that he had not given. Both corrected in this record.
- "assume i know nothing about the subject or the project how could i sill make you output better and qualitiy control it if possible": the AI proposed controlling AI output without subject knowledge. Gate only irreversible or outward steps, state checkable done-criteria, and ask what was not verified.
- "okay so i am sure we are the ai aided pair. meaning . could i just push without looking at hyour stuff ie the branch and thell him": the AI asked which items to push. Alexander's answer: "i would say i should be transparent with him about ai usage but i did not look at anystuff yet. so maybe just push somethin right now i tell him i loooked roughly what he did and then i do the rest on monday". Result: only the fixes branch (review/robustness) was pushed, on 2026-11-07. This record stayed local until 2026-11-09, and the optional branch is still local.
- "so why is it so bullet pointe": the WhatsApp drafts became short paragraphs without bullets.
- "or just details? considering the scope till wednesday": the proposed click-linked scatter was cut to the plain scatter: "just the plain scatter (no clicking, that can come later)".

------------

# 2026-11-11 – ChatGPT: Project Planning

- **Who:** Cristian Knebel
- **Tool:** ChatGPT (GPT-6)
- **Purpose:** Idea and choosing visualizations.

## Important prompt

> Give me a concrete project idea for "How do pretrained neural networks confuse similar classes?".

## Suggestions I chose and why

| Visualization / Method | Why I chose it |
|---|---|
| Language Shift | To investigate how changing the language or class descriptions affects the model's predictions and reveals differences in how classes are interpreted. |
| Prediction Explorer | To interactively explore model predictions, compare correct and incorrect classifications, and identify patterns in classification errors. |

I chose these methods because they investigate different aspects of the research question: Language Shift helps explore how changes in language affect predictions, while Prediction Explorer allows me to inspect individual predictions and identify patterns in classification errors.

## Other decisions

- **Model:** `openai/clip-vit-base-patch32`, a pretrained CLIP model that compares image and text representations to calculate image–text similarity.
- **Dataset:** `clip-benchmark/wds_country211`, a dataset containing images from 211 countries, used to investigate how CLIP recognizes and predicts country-related concepts.
- **Approach:** First analyze CLIP's predictions, then investigate how changing text descriptions affects the predictions (Language Shift). Use the Prediction Explorer to inspect individual predictions and identify patterns in classification errors.