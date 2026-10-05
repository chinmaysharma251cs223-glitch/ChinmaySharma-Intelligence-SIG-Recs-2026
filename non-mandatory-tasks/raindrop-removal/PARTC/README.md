# Part C: Restoring the Masked Regions

## Goal

Build a restoration model that takes a rainy image (and, optionally, the mask predicted in Part B) and produces a cleaned image, filling in the raindrop-affected regions using the rest of the image as context.

## What You Need to Do

1. Build a restoration/inpainting model. You can condition it on the mask from Part B (mask-guided restoration) or design it to restore the full image directly without an explicit mask input, your choice, but justify it.
2. Train it using the real clean images from the Raindrop Dataset as ground truth (this is why the paired dataset matters, you have genuine ground truth here, unlike the masks in Part B).
3. Evaluate your restored images against the real clean ground-truth images using PSNR and SSIM.

## Encouraged Experimentation

- Compare mask-guided restoration against a model that restores the whole image without an explicit mask
- Try different restoration architectures (autoencoder, GAN-based, diffusion-based) and compare quality and training cost
- Look at where restoration fails, large raindrop clusters, textured backgrounds, or regions near image borders

## Deliverables

- `/PartC/src/` with your restoration model code
- `/PartC/output/restored/` with restored images on the test set
- `/PartC/output/metrics.csv` with PSNR/SSIM per test image, plus the average
- `/PartC/README.md` documenting your architecture choice, training setup, quantitative results, and qualitative before/after examples (include at least 5 side-by-side comparisons: rainy, restored, ground truth)

## Notes

This part gives you a full rainy-to-clean pipeline for the first time (Part B locates the problem, Part C fixes it). Part C's output is close to what the Finale asks for, so build it in a way you can extend rather than throw away.