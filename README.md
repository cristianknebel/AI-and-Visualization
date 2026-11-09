# Austria or Australia? Testing CLIP's view of two countries
 
## Task
 
Austria and Australia are famously confused by name, but can an AI tell them apart just by looking? We use CLIP for zero-shot image classification on photos from both countries to find out whether it really recognizes the countries or has only learned their postcard stereotypes.
 
We compare different label sets, such as:
 
- `[Austria, Australia]`
- `[Österreich, Australien]`
- `[Österreich, Australia]`

We also group the photos by type (mountains, lakes, city sights, everyday scenes) to see where the errors concentrate.
 
## Model
 
**CLIP ViT-B/32** (OpenAI), pre-trained and used zero-shot, without any training.
 
- Model: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- Paper: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- CLIP documentation: [Hugging Face Transformers – CLIP](https://huggingface.co/docs/transformers/model_doc/clip)
- Hugging Face documentation: [huggingface.co/docs](https://huggingface.co/docs)
## Datasets
 
| Role | Dataset | Description | Link |
|---|---|---|---|
| Main | Country211 | Austria and Australia photos (geotagged Flickr photos, 300 per country) | [country211.md](https://github.com/openai/CLIP/blob/main/data/country211.md) |
| Optional | OSV-5M | Street-view photos for everyday scenes | [osv5m/osv5m](https://huggingface.co/datasets/osv5m/osv5m) |
| Optional | Own tricky collection | Hand-picked hard cases (e.g. Neusiedler See, snowy Australian mountains) | – |
