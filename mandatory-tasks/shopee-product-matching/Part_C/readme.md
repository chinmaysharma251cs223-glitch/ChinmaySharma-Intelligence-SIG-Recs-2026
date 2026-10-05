# Part C - Image-Based Product Matching

## Objective

Product titles are only one source of information available in the Shopee dataset.

In this section, you will investigate whether **visual information** can be used to identify matching products.

Your objective is to develop an image-based product matching system.

---

## Task

Build a system that compares two product images and determines how likely they are to represent the same underlying product.

A general solution may follow a pipeline such as:

```text
Product Image
      ↓
Image Representation
      ↓
Feature / Embedding
      ↓
Similarity
      ↓
Matching Decision
```

The exact approach is up to you.

---

## Baseline

Start with a pretrained computer vision model or another reasonable image representation method.

You may investigate approaches such as:

* CNN-based models
* ResNet
* EfficientNet
* Vision Transformers
* CLIP
* Other pretrained vision models

You are not restricted to these approaches.

---

## Image Embeddings

Investigate the concept of image embeddings.

Your analysis should demonstrate an understanding of:

* What an embedding represents
* Why embeddings can be used for similarity search
* How two image embeddings can be compared
* How similarity scores can be converted into matching decisions

---

## Experiments

Experiment with at least **two approaches or configurations**.

For each experiment, report:

* Model/representation used
* Embedding dimension
* Similarity measure
* Threshold
* Computational considerations
* Matching performance

For example:

| Experiment   | Model | Similarity | Threshold | Score |
| ------------ | ----- | ---------- | --------- | ----- |
| Baseline     | ...   | ...        | ...       | ...   |
| Experiment 1 | ...   | ...        | ...       | ...   |
| Experiment 2 | ...   | ...        | ...       | ...   |

---

## Nearest-Neighbor Analysis

For selected products, retrieve the most visually similar products from the dataset.

For example:

```text
Query Image
     ↓
Image Embedding
     ↓
Similarity Search
     ↓
Top-K Similar Products
```

Visualize the retrieved results and analyze whether the nearest neighbors actually correspond to the same product.

---

## Error Analysis

Investigate cases such as:

* Same product photographed differently
* Different products with similar appearance
* Different colors or variants
* Different packaging
* Different backgrounds
* Cropped images
* Low-quality images

Identify situations where your approach fails and explain why.

---

## Deliverables

Submit:

```text
PartC/
├── notebook.ipynb
├── README.md
└── results/
```

Your submission should include:

* Image preprocessing
* Baseline
* Experiments
* Similarity analysis
* Nearest-neighbor examples
* Error analysis
* Conclusions

---

## Questions to Consider

> What information does an image embedding capture?

> Why might two images of the same product have different embeddings?

> Why might two different products have highly similar embeddings?

> Which similarity metric works best for your representation?

> How does the matching threshold affect your results?

> What are the computational challenges of comparing a large number of images?

---

## Evaluation

Your work will be evaluated based on:

* Understanding of image representations
* Quality of experimentation
* Interpretation of similarity
* Error analysis
* Computational awareness
* Ability to justify design decisions

You are not expected to train a vision model from scratch.

The focus is on **understanding and effectively using visual representations for the matching problem**.
