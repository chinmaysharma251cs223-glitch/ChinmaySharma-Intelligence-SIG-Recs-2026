# Finale: End-to-End Raindrop Removal Pipeline

## Task Overview

This is the culmination of Parts A, B, and C. You've built features, a segmentation model, and a restoration model. In the Finale, you'll design and implement a complete pipeline, rainy image in, clean image out, and push it further than the basic Part B + Part C combination.

## Part 1: Design

Write a short design document (1-2 pages) addressing the following:

- How does your pipeline combine segmentation and restoration? Are they two separate stages (like Parts B and C), or do you propose an end-to-end model that does both jointly?
- **The hard case:** how would you handle heavy rain, where raindrops overlap and cover a large fraction of the image, so masking alone loses too much information for restoration to work well? Propose a concrete approach (e.g. using multiple frames if available, leveraging stronger priors, a different architecture entirely).
- How would you evaluate your pipeline on real-world rainy images, where you don't have a clean ground-truth counterpart to compute PSNR/SSIM against?
- What you'd change or add with more time/resources.

## Part 2: Implementation

Implement your designed pipeline. At minimum, it should:

1. Improve on or meaningfully differ from simply chaining your Part B and Part C models as-is, implement whatever idea you proposed for the hard case above, even in a simplified form.
2. Run end to end on the full Raindrop Dataset test set (58 pairs) and report PSNR/SSIM against ground truth.
3. Include a qualitative test on at least 3 images outside the Raindrop Dataset (any rainy photo you find or take yourself) to see how it generalizes, no ground truth needed here, just show the before/after.

## Deliverables

- `/Finale/design.md` with your design document
- `/Finale/src/` with your implementation
- `/Finale/output/test_set_results/` with restored images and metrics on the official 58-pair test set
- `/Finale/output/generalization_test/` with before/after results on your own out-of-dataset images
- `/Finale/README.md` summarizing what you built, your final PSNR/SSIM numbers compared to your Part C baseline, what worked, what didn't, and what you'd do differently with more time

## Evaluation

We're looking at:

- Whether your pipeline genuinely improves on or thoughtfully differs from the naive Part B + Part C chain
- How well you've addressed the heavy-rain/overlapping-raindrops case, even a partial or simplified solution with good reasoning counts
- Quantitative results (PSNR/SSIM) on the test set, and qualitative results on your own images
- How well you've documented tradeoffs and limitations

Novel ideation and smart experimentation are valued over a "safe" pipeline that technically works but doesn't show much thought.