# Part A - Dataset Exploration

## Objective

Before building a product matching system, it is important to understand the data you are working with.

In this section, your objective is to perform an exploratory analysis of the Shopee Product Matching dataset and identify the characteristics that make product matching challenging.

---

## Tasks

### 1. Understand the Dataset

Explore the available files and columns.

Determine:

* What information is available for each listing?
* What does each column represent?
* What constitutes a product group?
* How is the matching relationship represented?

---

### 2. Analyze the Dataset

Investigate properties such as:

* Number of product listings
* Number of unique product groups
* Distribution of product-group sizes
* Number of unique images
* Duplicate images
* Duplicate or highly similar titles
* Distribution of title lengths
* Other interesting properties you discover

Present important findings using appropriate visualizations.

---

### 3. Investigate Product Similarity

Find and visualize examples of:

* Multiple listings representing the same product
* Products with different images but the same product identity
* Products with similar titles but different product identities
* Products with visually similar images but different product identities
* Listings with noisy or incomplete information

For each example, explain what makes the matching problem difficult.

---

### 4. Identify Challenges

Based on your analysis, identify the major challenges that a machine learning model may encounter.

For example:

* Noisy product titles
* Different languages
* Abbreviations
* Visually similar products
* Different image backgrounds
* Missing information
* Variations in product presentation

You are encouraged to identify challenges beyond these examples.

---

## Deliverables

Submit a notebook containing:

* Dataset exploration
* Statistical analysis
* Visualizations
* Example product groups
* Identified challenges
* Your observations and conclusions

---

## Questions to Consider

Your analysis should help you answer questions such as:

> What makes two listings belong to the same product?

> Can product titles alone reliably determine whether two listings match?

> Can images alone reliably determine whether two listings match?

> What types of examples are likely to be difficult for a machine learning model?

> What information in the dataset could potentially be useful for solving the problem?

You do not need to build the final matching model in this section.

The purpose of this section is to **understand the problem before attempting to solve it**.

---

## Evaluation

This section will primarily be evaluated on:

* Quality of exploration
* Understanding of the dataset
* Relevance of visualizations
* Ability to identify meaningful patterns
* Quality of observations

Simply generating plots without explaining what they reveal will not be considered sufficient.

---

## Submission

Place your work inside:

```text
PartA/
├── notebook.ipynb
└── README.md
```

If additional files are required, organize them appropriately and document them in your README.
