# CrossModal Alignment

## Vision-Language Representation Learning

### Problem Statement

Image and text contain different forms of information, yet many real-world applications require models to reason across both modalities.

Your task is to build a **vision-language representation model from scratch** that learns a shared embedding space for images and text.

Given a matching image-text pair, the learned representations should be close. Given unrelated pairs, they should be separated.

The resulting model should support both **image-to-text** and **text-to-image retrieval**, and should demonstrate that the learned representation captures meaningful semantic relationships rather than dataset-specific shortcuts.

You are free to choose the model architectures, representation methods, training strategy, and experimental design.

---

## Part 1 - Dataset & Representation

Use a paired image-text dataset to construct your training and evaluation data.

### Dataset

**Flickr8k**

* 8,092 images
* 5 human-written captions per image
* Standard train/validation/test splits

Dataset information: https://www.kaggle.com/datasets/dibyansudiptiman/flickr-8k

Alternative dataset:
* **MS COCO** - https://cocodataset.org/

Define appropriate training, validation, and test splits while ensuring that images do not leak across splits. You are free to use any dataset.

You must document your preprocessing and representation choices.

### Resources

* **Learning Transferable Visual Models From Natural Language Supervision (CLIP)** https://arxiv.org/abs/2103.00020
* **A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)** https://arxiv.org/abs/2002.05709
* **ALIGN: Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision** https://arxiv.org/abs/2102.05918

These resources are provided to understand the underlying research problem. **Do not use a pretrained CLIP model as your solution.**

--

## Part 2 - Cross-Modal Alignment

Develop a **contrastive training objective** that learns a shared representation for images and text.

For each batch, the model should learn to assign high similarity to matching image-text pairs and low similarity to mismatched pairs.

Investigate how different **negative-sampling strategies** and **training configurations** affect the quality of the learned alignment.

Evaluate your choices quantitatively and justify your design decisions.


---

## Part 3 - Retrieval

Evaluate the learned representation in both directions.

### Image → Text

Given an image, retrieve the most relevant captions from a candidate set.

### Text → Image

Given a caption, retrieve the most relevant images.

Report:

* Recall@1
* Recall@5
* Recall@10

Analyse representative successful and failed retrievals.

The objective is not only to report retrieval scores, but to identify **what semantic relationships the model captures and where it fails**.

---

### Part 4 - Generalization & Analysis

Test whether the model works beyond the image-text pairs it was trained on.

Evaluate at least three of the following:

Zero-shot classification using text descriptions Hard-negative retrieval Image corruptions Text changes Compositional understanding Distribution shifts Embedding-space visualization Spurious correlations

You must also design one experiment of your own to investigate a question about the model.

For each experiment, include:

Hypothesis - what you expect to happen Setup - how you test it Results - what you observe Analysis - what the results tell you

Your final report should compare the different experiments, include retrieval and generalization results, show important failure cases, and discuss the limitations of your model.

The analysis should ultimately answer:

Has the model learned meaningful relationships between images and text, or is it relying on simple patterns in the dataset?

### Submission Source code/colab Notebook

Technical report covering: Model architecture and design choices, Evaluation methodology and experiments, Results and relevant visualizations, Analysis of the results and observed limitations.

