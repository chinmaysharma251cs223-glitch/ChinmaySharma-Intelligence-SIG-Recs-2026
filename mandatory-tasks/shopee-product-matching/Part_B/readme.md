# Part B - Text-Based Product Matching

## Objective

In this section, you will attempt to solve the product matching problem using **textual information**.

Product listings contain titles that may provide useful information about product identity. However, these titles can also contain noise, abbreviations, spelling variations, multiple languages, seller-specific terminology, and irrelevant information.

Your goal is to investigate how effectively textual information can be used for product matching.

---

## Task

Build a system that takes two product listings and determines whether they correspond to the same underlying product using their textual information.

Your system should follow the general pipeline:

```text
Product Titles
      ↓
Text Representation
      ↓
Similarity / Matching
      ↓
Prediction
```

The exact implementation is up to you.

---

## Baseline

Begin with a simple baseline approach.

You should establish a measurable baseline before experimenting with more sophisticated methods.

Possible techniques include:

* Bag of Words
* TF-IDF
* Word n-grams
* Character n-grams
* Word embeddings
* Sentence embeddings
* Transformer-based representations

These are suggestions, not requirements.

---

## Experiments

Experiment with at least **two different approaches** for representing or comparing product titles.

For each experiment, document:

* Method used
* Why you chose it
* How it was implemented
* Evaluation metric
* Results
* Observations

A useful experiment table could look like:

| Experiment   | Representation | Similarity | Threshold | Score |
| ------------ | -------------- | ---------- | --------- | ----- |
| Baseline     | TF-IDF         | Cosine     | ...       | ...   |
| Experiment 1 | ...            | ...        | ...       | ...   |
| Experiment 2 | ...            | ...        | ...       | ...   |

---

## Threshold Analysis

If your approach produces a similarity score, investigate how the matching threshold affects the results.

Analyze the trade-off between:

* Precision
* Recall
* False positives
* False negatives

Explain how you selected your final threshold.

---

## Error Analysis

Identify examples where your model:

* Correctly predicts a match
* Incorrectly predicts a match
* Fails to identify a true match

Analyze why these errors occur.

For example:

```text
Title A:
"Samsung Galaxy Buds Pro Original"

Title B:
"Samsung Wireless Earbuds Pro"

Prediction:
Match

Was the prediction correct?
Why?
```

Use actual examples from your experiments where possible.

---

## Deliverables

Submit:

```text
PartB/
├── notebook.ipynb
├── README.md
└── results/
```

Your notebook should contain:

* Data preparation
* Baseline implementation
* Experiments
* Evaluation
* Threshold analysis
* Error analysis
* Conclusions

---

## Questions to Consider

Think about the following:

> How should two similar product titles be represented mathematically?

> What makes TF-IDF effective or ineffective for this problem?

> Would character-level features be useful?

> Why might semantic embeddings outperform keyword-based approaches?

> What happens when two different products have very similar titles?

> What happens when the same product has completely different titles?

You do not need to answer every question explicitly, but your experiments should demonstrate your understanding of the underlying problem.

---

## Evaluation

Evaluation will consider:

* Quality of the baseline
* Understanding of text representations
* Experimental methodology
* Metric interpretation
* Error analysis
* Reasoning behind design choices

A more complex model is **not automatically a better solution**.

A simple approach accompanied by strong reasoning and experimentation is preferable to a sophisticated approach that is not understood.
