"""
Grad-CAM for CLIP: which parts of a photo make CLIP say "Austria" or "Australia"?

For every photo it saves one image with three panels:
  1. the photo as CLIP sees it (224 x 224 centre crop),
  2. a heatmap of the regions that raise CLIP's score for the first label,
  3. the same for the second label.
Red = this region pushes CLIP towards that label, blue = little influence.

Reads the photos from cache_at_au (written by Prediction explorer.py) and
saves the images to gradcam_output/<labels>/, next to this script.
Photos that already have an image are skipped, so you can stop and continue later.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # only save files, don't open windows

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# Settings
CACHE = Path(r"C:\Users\crist\Desktop\JKU courses\Semester 5\AI-and-Visualization\Pair 1\AI4Vis\cache_at_au")
MODEL_NAME = "openai/clip-vit-base-patch32"
TEMPLATE = "a photo i took in {}."
LABELS = ["Austria", "Australia"]   # or ["Österreich", "Australien"], ["Österreich", "Australia"]
FILES = []                          # empty = all photos; or e.g. ["Austria_test_000.jpg"]
COLORS = {"Austria": "#c8102e", "Australia": "#1f4e9e"}
OUT_DIR = Path(__file__).parent / "gradcam_output" / "_".join(LABELS)


def add_gradcam_hook(model):
    """Remember the activations of the last vision layer and their gradients."""
    store = {}

    def hook(_module, _inputs, output):
        store["acts"] = output
        output.register_hook(lambda grad: store.__setitem__("grads", grad))

    model.vision_model.encoder.layers[-1].layer_norm1.register_forward_hook(hook)
    return store


def gradcam(model, proc, store, image, label_index):
    """Heatmap (224 x 224, values 0..1) for one label, plus CLIP's probabilities."""
    texts = [TEMPLATE.format(label) for label in LABELS]
    inputs = proc(text=texts, images=image, return_tensors="pt", padding=True)

    model.zero_grad()
    out = model(**inputs)
    out.logits_per_image[0, label_index].backward()   # gradient of this label's score

    acts = store["acts"][0, 1:]     # patch tokens (drop the CLS token): (49, 768)
    grads = store["grads"][0, 1:]
    weights = grads.mean(dim=0)     # how important each feature is for this label
    cam = torch.relu(acts @ weights)

    side = int(cam.numel() ** 0.5)  # 49 patches -> 7 x 7 grid
    cam = cam.reshape(1, 1, side, side)
    size = inputs["pixel_values"].shape[-2:]
    cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)

    probs = out.logits_per_image.softmax(dim=-1)[0].tolist()
    return cam.detach().numpy(), probs, inputs["pixel_values"][0]


def to_image(pixel_values, proc):
    """Undo CLIP's normalisation, so we show exactly what the model saw."""
    mean = torch.tensor(proc.image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(proc.image_processor.image_std).view(3, 1, 1)
    return (pixel_values * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def pick_photos():
    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))

    if FILES:
        return [m for m in meta if m["file"] in FILES]

    return meta


def plot_photo(model, proc, store, m):
    """One figure: the photo and one heatmap per label."""
    image = Image.open(CACHE / "images" / m["file"]).convert("RGB")
    cams = [gradcam(model, proc, store, image, i) for i in range(len(LABELS))]
    probs, pixels = cams[0][1], cams[0][2]
    seen = to_image(pixels, proc)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    axes[0].imshow(seen)
    axes[0].set_title(f"True: {m['country']}\n{m['file']}", fontsize=10, color=COLORS[m["country"]], fontweight="bold")

    for ax, label, (cam, _, _), p in zip(axes[1:], LABELS, cams, probs):
        ax.imshow(seen)
        ax.imshow(cam, cmap="jet", alpha=0.45)
        winner = p == max(probs)
        ax.set_title(f"Where CLIP sees '{label}'\n{p:.0%}", fontsize=11, fontweight="bold" if winner else "normal")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    model = CLIPModel.from_pretrained(MODEL_NAME).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_NAME)
    store = add_gradcam_hook(model)

    photos = pick_photos()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving Grad-CAM images for {len(photos)} photos to: {OUT_DIR}", flush=True)

    for i, m in enumerate(photos, start=1):
        out = OUT_DIR / f"gradcam_{Path(m['file']).stem}.png"

        if out.exists():
            continue   # already done in an earlier run

        fig = plot_photo(model, proc, store, m)
        fig.savefig(out, dpi=120)
        plt.close(fig)   # free the memory, otherwise 500 figures pile up

        if i % 25 == 0 or i == len(photos):
            print(f"  {i}/{len(photos)} done", flush=True)

    print("Finished.", flush=True)