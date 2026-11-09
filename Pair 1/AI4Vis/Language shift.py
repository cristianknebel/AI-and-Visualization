"""
Language-shift scatter: does switching to German country names change CLIP's mind?

One dot per photo. x = Austria score with English names (Austria / Australia),
y = Austria score with German names (Österreich / Australien).
Dots below the diagonal got a lower Austria score in German.

Reads the cache written by Prediction explorer.py (cache_at_au/meta.json and
cache_at_au/predictions.json), therefore, that script needs to run first. No model needed here.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

CACHE = Path("cache_at_au")
EN_KEY = "EN: Austria / Australia"
DE_KEY = "DE: Österreich / Australien"
COLORS = {"Austria": "#c8102e", "Australia": "#1f4e9e"}
MARKERS = {"Austria": "o", "Australia": "^"}
SAVE_AS = "language_shift_scatter.png"   # set to None to only show the window

# Data
meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
preds = json.loads((CACHE / "predictions.json").read_text(encoding="utf-8"))

# index 0 of every label set is the Austria score
en = [p["sets"][EN_KEY][0] for p in preds]
de = [p["sets"][DE_KEY][0] for p in preds]
truth = [m["country"] for m in meta]

n = len(preds)
n_below = sum(y < x for x, y in zip(en, de))
n_flipped = sum(x >= 0.5 and y < 0.5 for x, y in zip(en, de))

# Plot
fig, ax = plt.subplots(figsize=(8, 7.4))
fig.patch.set_facecolor("#fbfbf9")
ax.set_facecolor("#fbfbf9")

# grey area: English says Austria, German says Australia
ax.fill_between([0.5, 1.02], -0.02, 0.5, color="#ebebeb", zorder=0)
ax.text(0.02, 0.80, f"Grey area: English says Austria,\nGerman says Australia ({n_flipped} photos)",
        fontsize=10, va="center", color="#333")

# decision thresholds and diagonal
ax.axvline(0.5, color="#d0d0d0", lw=0.8, zorder=1)
ax.axhline(0.5, color="#d0d0d0", lw=0.8, zorder=1)
ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1, zorder=1)
ax.text(0.40, 0.42, "same score in both languages", rotation=45, rotation_mode="anchor",
        ha="center", va="bottom", fontsize=9, color="#777", transform_rotates_text=True)

# dots
for country in ["Austria", "Australia"]:
    xs = [x for x, t in zip(en, truth) if t == country]
    ys = [y for y, t in zip(de, truth) if t == country]
    ax.scatter(xs, ys, s=38, marker=MARKERS[country], color=COLORS[country],
               edgecolor="white", linewidth=0.6, zorder=3,
               label=f"{country} photo ({len(xs)})")

# axes
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")
ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_xlabel("CLIP's score for Austria with English names (Austria / Australia)", fontsize=11)
ax.set_ylabel("CLIP's score for Austria with German names (Österreich / Australien)", fontsize=11)
ax.tick_params(colors="#555", labelsize=10)

for side in ["top", "right"]:
    ax.spines[side].set_visible(False)

for side in ["left", "bottom"]:
    ax.spines[side].set_color("#bbb")

ax.legend(loc="upper left", frameon=False, fontsize=10)

# title: headline + what to read
fig.text(0.08, 0.955, "German country names push CLIP towards Australia",
         fontsize=15, color="#222")
fig.text(0.145, 0.92, f"One dot per photo. Dots below the dashed line got a lower Austria "
                      f"score in German: {n_below} of {n}.", fontsize=10, color="#666")

plt.subplots_adjust(top=0.89, bottom=0.09, left=0.1, right=0.97)

if SAVE_AS:
    out = Path(__file__).parent / SAVE_AS
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved plot to: {out}", flush=True)

plt.show()