# The Visionaires: Austria or Australia?

JKU 364.036 AI and Visualization, 2026W.

## Task

Zero-shot image classification with CLIP. Can it tell Austria and Australia apart from photos, or has it only learned their postcard stereotypes? And how much does its answer depend on the country names it is given (Austria vs Österreich), although CLIP is officially meant for English only?

## Model

CLIP ViT-B/32 by OpenAI, pre-trained and used zero-shot, without any training: https://huggingface.co/openai/clip-vit-base-patch32 (paper: https://arxiv.org/abs/2103.00020)

## Dataset

Country211 by OpenAI, geotagged Flickr photos from 211 countries: https://github.com/openai/CLIP/blob/main/data/country211.md

We stream it from the Hugging Face copy `clip-benchmark/wds_country211` and use the 100 Austria and 100 Australia photos of its test split. The photos come from the YFCC100M Flickr collection and are under Creative Commons licenses.

## Repository

- `Pair 1/AI4Vis/Visualizations.py`: prediction explorer. For each photo it shows CLIP's answer with English, German and mixed country names, plus its top 5 of all 211 countries.
- `Pair 2/`: the human-only pair of Phase I.
- `ai-interactions/`: record of the AI-aided pair's important prompts.

## Run

```
pip install -r requirements.txt
python "Pair 1/AI4Vis/Visualizations.py"
```

The photos and predictions are cached in `Pair 1/AI4Vis/cache_at_au`, so the explorer opens without downloading anything. Without the cache, the first run downloads the CLIP weights (about 1.2 GB, as transformers fetches two copies) and about 210 MB of the dataset.
