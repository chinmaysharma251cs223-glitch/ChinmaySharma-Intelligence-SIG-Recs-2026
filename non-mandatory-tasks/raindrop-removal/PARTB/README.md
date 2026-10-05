# Part B: Segmenting Raindrop Regions

## Goal

Build a segmentation model that predicts a mask of the raindrop-affected regions in a rainy image.

## Important: this dataset has no ground-truth masks

The Raindrop Dataset only gives you paired rainy/clean images, not manual segmentation masks. You need to derive your own pseudo-masks before you can train/evaluate a segmentation model. A standard approach (used in the original paper this dataset comes from):

1. For each rainy/clean pair, compute the pixel-wise difference between the two images.
2. Threshold that difference map to get a binary mask (raindrop region vs. clear region). You'll likely need some smoothing/morphological cleanup, since raw thresholded differences tend to be noisy.
3. Treat this as your pseudo ground-truth mask for training and evaluating your segmentation model.

You're free to improve on this basic approach, better thresholding, edge-aware refinement, or a different pseudo-labeling strategy entirely, as long as you explain and justify it.

## What You Need to Do

1. Generate pseudo-masks for the training set using the approach above (or your own improved version). Save these to `/PartB/pseudo_masks/`.
2. Train a segmentation model (a lightweight U-Net or similar is enough, doesn't need to be state-of-the-art) that predicts a raindrop mask given only the rainy image (no clean image at inference time, that's the whole point, you won't have it in the real world).
3. Evaluate your segmentation model on the test set by comparing predicted masks against pseudo-masks derived the same way from the test pairs.

## Encouraged Experimentation

- Compare a few different pseudo-masking strategies and show how much they affect your final segmentation quality
- Try incorporating the Part A embeddings/features as auxiliary input to the segmentation model
- Look at failure cases, does your model struggle with small raindrops, large ones, or ones near image edges

## Deliverables

- `/PartB/pseudo_masks/` with your derived masks for train and test sets
- `/PartB/src/` with your pseudo-masking code and your segmentation model code
- `/PartB/output/predictions/` with predicted masks on the test set
- `/PartB/README.md` documenting your pseudo-masking approach, your segmentation model, evaluation results, and observed failure cases

## Notes

Be upfront in your README about the fact that your "ground truth" here is itself derived, not manually annotated, and discuss how that might limit what your segmentation model can learn.