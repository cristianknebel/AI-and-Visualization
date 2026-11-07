import json
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, CheckButtons
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import CLIPModel, CLIPProcessor

# Settings
REPO = "clip-benchmark/wds_country211"
SPLIT = "test"  # 100 photos per country in the test split
N_PER_COUNTRY = 100  # lower this (e.g. 30) for a faster first run
MODEL_NAME = "openai/clip-vit-base-patch32"
TEMPLATE = "a photo i took in {}."
CACHE = Path("cache_at_au")
BATCH = 16

# Every label set lists Austria first, Australia second.
LABEL_SETS = {
    "EN: Austria / Australia": ["Austria", "Australia"],
    "DE: Österreich / Australien": ["Österreich", "Australien"],
    "Mixed: Österreich / Australia": ["Österreich", "Australia"],
}
COUNTRIES = ["Austria", "Australia"]
COLORS = {"Austria": "#c8102e", "Australia": "#1f4e9e"}
CLOSE_CALL = 0.20  # margin below which a decision counts as a close call


# Data
def load_classnames():
    path = hf_hub_download(REPO, "classnames.txt", repo_type="dataset")

    with open(path, encoding="utf-8") as file:
        names = [line.strip() for line in file if line.strip()]
    for country in COUNTRIES:
        if country not in names:
            raise ValueError(f"'{country}' not found in classnames.txt: {names[:10]} ...")

    return names


def collect_images(classnames):
    """Stream the split once and save the Austrian/Australian photos locally."""
    img_dir, meta_path = CACHE / "images", CACHE / "meta.json"

    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    img_dir.mkdir(parents=True, exist_ok=True)
    wanted = {classnames.index(country): country for country in COUNTRIES}
    counts = {country: 0 for country in COUNTRIES}
    meta = []

    dataset = load_dataset(REPO, split=SPLIT, streaming=True)
    print("Scanning the dataset for Austria/Australia photos...")

    for index, sample in enumerate(dataset):
        cls = int(sample["cls"])
        name = wanted.get(cls)

        if name and counts[name] < N_PER_COUNTRY:
            img = next(value for value in sample.values() if isinstance(value, Image.Image))
            fname = f"{name}_{counts[name]:03d}.jpg"
            img.convert("RGB").save(img_dir / fname, quality=92)
            meta.append({"file": fname, "country": name})
            counts[name] += 1

        if all(value >= N_PER_COUNTRY for value in counts.values()):
            break

        if index % 2000 == 0:
            print(f"  scanned {index:>6} samples, found {counts}")

    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Saved {len(meta)} photos to {img_dir}")
    return meta


# Model
@torch.no_grad()
def compute_predictions(meta, classnames):
    pred_path = CACHE / "predictions.json"

    if pred_path.exists():
        return json.loads(pred_path.read_text(encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_NAME)

    # texts we need: the small label sets + all 211 countries
    set_names = [n for names in LABEL_SETS.values() for n in names]
    texts = [TEMPLATE.format(n) for n in set_names + classnames]

    results = []

    for start in range(0, len(meta), BATCH):
        chunk = meta[start:start + BATCH]
        images = [Image.open(CACHE / "images" / m["file"]).convert("RGB") for m in chunk]
        inputs = proc(text=texts, images=images, return_tensors="pt", padding=True).to(device)
        logits = model(**inputs).logits_per_image  # (batch, n_texts): 100 x cosine similarity

        for row in logits:
            entry, pos = {"sets": {}}, 0

            for key, names in LABEL_SETS.items():
                entry["sets"][key] = row[pos:pos + len(names)].softmax(-1).tolist()
                pos += len(names)

            p211 = row[pos:].softmax(-1)
            top = p211.topk(5)
            entry["top211"] = [[classnames[i], p] for p, i in zip(top.values.tolist(), top.indices.tolist())]
            results.append(entry)

        print(f"  predicted {min(start + BATCH, len(meta))}/{len(meta)}")

    pred_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results


# Viewer
def truth_index(m):
    return COUNTRIES.index(m["country"])


def is_mistake(m, p):
    t = truth_index(m)
    return any(probs.index(max(probs)) != t for probs in p["sets"].values())


def explorer(meta, preds):
    all_idx = list(range(len(meta)))
    state = {"view": all_idx, "pos": 0}

    fig = plt.figure(figsize=(14, 8))
    fig.canvas.manager.set_window_title("CLIP prediction explorer: Austria vs. Australia")
    gs = fig.add_gridspec(4, 2, width_ratios=[1.25, 1], hspace=1.0, wspace=0.15,
                          left=0.03, right=0.97, top=0.90, bottom=0.12)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_sets = [fig.add_subplot(gs[i, 1]) for i in range(len(LABEL_SETS))]
    ax_top = fig.add_subplot(gs[3, 1])

    def draw():
        for ax in [ax_img, *ax_sets, ax_top]:
            ax.clear()

        view = state["view"]

        if not view:
            ax_img.axis("off")
            ax_img.text(0.5, 0.5, "No images match this filter", ha="center", va="center")
            fig.canvas.draw_idle()
            return

        j = view[state["pos"]]
        m, p = meta[j], preds[j]
        t = truth_index(m)

        # image, with the parts CLIP never sees shaded: it keeps only the centre square
        img = Image.open(CACHE / "images" / m["file"])
        ax_img.imshow(img)
        w, h = img.size
        s = min(w, h)
        for x, y, dx, dy in [(0, 0, (w - s) / 2, h), ((w + s) / 2, 0, (w - s) / 2, h),
                             (0, 0, w, (h - s) / 2), (0, (h + s) / 2, w, (h - s) / 2)]:
            ax_img.add_patch(Rectangle((x - 0.5, y - 0.5), dx, dy, color="black", alpha=0.55, linewidth=0))
        ax_img.axis("off")
        ax_img.text(0.5, -0.01, "shaded: cut off before CLIP sees the photo", transform=ax_img.transAxes,
                    ha="center", va="top", fontsize=9, color="#555555")
        ax_img.set_title(f"True country: {m['country']}   ({m['file']})",
                         color=COLORS[m["country"]], fontsize=13, fontweight="bold")

        # one bar chart per label set
        for ax, (key, probs) in zip(ax_sets, p["sets"].items()):
            names = LABEL_SETS[key]
            colors = [COLORS[c] for c in COUNTRIES]
            ax.barh(names[::-1], probs[::-1], color=colors[::-1])
            ax.set_xlim(0, 1)
            ax.tick_params(labelsize=10)

            for y, v in enumerate(probs[::-1]):
                ax.text(min(v + 0.01, 0.88), y, f"{v:.1%}", va="center", fontsize=10)

            pred = probs.index(max(probs))
            margin = abs(probs[0] - probs[1])
            verdict = "✓ correct" if pred == t else "✗ wrong"
            kind = "close call" if margin < CLOSE_CALL else "confident"
            ax.set_title(f"{key}   {verdict}, {kind} (margin {margin:.0%})",
                         fontsize=10, loc="left",
                         color="#1a7f37" if pred == t else "#b42318")

        # top-5 over all 211 countries
        names = [n for n, _ in p["top211"]][::-1]
        vals = [v for _, v in p["top211"]][::-1]
        colors = [COLORS[m["country"]] if n == m["country"] else "#9aa0a6" for n in names]
        ax_top.barh(names, vals, color=colors)
        ax_top.set_xlim(0, 1)
        ax_top.tick_params(labelsize=10)

        for y, v in enumerate(vals):
            ax_top.text(min(v + 0.01, 0.88), y, f"{v:.1%}", va="center", fontsize=10)

        ax_top.set_title("Top 5 of all 211 countries (true country coloured)",
                         fontsize=10, loc="left")

        n_err = sum(is_mistake(meta[i], preds[i]) for i in all_idx)
        fig.suptitle(f"Image {state['pos'] + 1} of {len(view)}   ·   "
                     f"{n_err}/{len(all_idx)} images wrong in at least one label set",
                     fontsize=11)
        fig.canvas.draw_idle()

    def step(delta):
        if state["view"]:
            state["pos"] = (state["pos"] + delta) % len(state["view"])
            draw()

    def toggle(_label):
        only_err = chk.get_status()[0]
        state["view"] = [i for i in all_idx if is_mistake(meta[i], preds[i])] if only_err else all_idx
        state["pos"] = 0
        draw()

    b_prev = Button(fig.add_axes([0.30, 0.025, 0.08, 0.05]), "◀ Prev")
    b_next = Button(fig.add_axes([0.39, 0.025, 0.08, 0.05]), "Next ▶")
    chk = CheckButtons(fig.add_axes([0.50, 0.02, 0.14, 0.06]), ["Only mistakes"], [False])
    b_prev.on_clicked(lambda _e: step(-1))
    b_next.on_clicked(lambda _e: step(+1))
    chk.on_clicked(toggle)
    fig.canvas.mpl_connect("key_press_event", lambda e: step(-1) if e.key == "left" else step(1) if e.key == "right" else None)

    # open the window maximized
    manager = plt.get_current_fig_manager()

    try:
        manager.window.state("zoomed")        # TkAgg (default on Windows)
    except AttributeError:
        try:
            manager.window.showMaximized()    # Qt backends
        except AttributeError:
            pass                              # other backends: leave as is

    draw()
    plt.show()


if __name__ == "__main__":
    classnames = load_classnames()
    meta = collect_images(classnames)
    preds = compute_predictions(meta, classnames)
    explorer(meta, preds)