# Part A: Building Image Features/Embeddings

## Goal

The goal of this part is to experiment with representing images from the Raindrop Dataset as embeddings/features. These representations will be useful context for Part B (finding where the raindrops are) and give you an early feel for how rainy vs. clean images differ in feature space.

## Dataset

The Raindrop Dataset (Qian et al., CVPR 2018) is  split into `train/` and `test_a/`, each containing paired `rain/` and `clean/` image folders (matching filenames across the two). Everyone must use this dataset for Parts A, B, and C so results are comparable across submissions.

dataset link : https://drive.google.com/drive/folders/1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K?usp=sharing

## What You Need to Do

1. Extract image-level (and optionally patch-level) embeddings for both the rainy and clean images, using a pretrained backbone (e.g. a CNN or ViT feature extractor) or one you train yourself.
2. Save your embeddings as a CSV with (image_id, split [rain/clean], embedding) columns and upload it to `/PartA/output/`.
3. Do a basic comparison: do rainy and clean versions of the same scene sit close together or far apart in embedding space? Visualize this (e.g. with t-SNE/PCA) and note what you observe.

## Encouraged Experimentation

- Try patch-level embeddings (splitting the image into a grid) in addition to whole-image embeddings, these can be more useful later for localizing raindrop regions
- Compare a pretrained backbone against a self-supervised or fine-tuned one
- Look at whether embedding distance between a rainy image and its clean pair correlates with how heavily that image is affected by raindrops

Novel ideation and smart experimentation are highly appreciated, more so than squeezing out marginal performance gains.

## Deliverables

- `/PartA/output/embeddings.csv` with (image_id, split, embedding) pairs
- `/PartA/output/visualization.png` (or similar) showing your embedding space comparison
- `/PartA/README.md` documenting:
  - Your feature extraction approach and why you chose it
  - What you observed comparing rainy vs. clean embeddings
  - Any experiments you ran

## Notes

Keep your code in `/PartA/src/` and make sure it can be re-run end to end (raw dataset in, CSV + visualization out) without manual steps.