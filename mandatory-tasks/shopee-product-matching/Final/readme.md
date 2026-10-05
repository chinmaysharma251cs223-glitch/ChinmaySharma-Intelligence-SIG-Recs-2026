# Finale - Multimodal Product Matching

## Objective

You have now explored the Shopee Product Matching problem from both textual and visual perspectives.

Your final task is to build a **complete product matching system** using the knowledge gained from the previous sections.

The objective is not simply to combine the previous solutions.

You should investigate how different sources of information can work together and develop a solution that you believe is appropriate for the problem.

## Problem

Given product listings, identify which listings correspond to the same underlying product.

Your system may make use of:

- Product images
- Product titles
- Image representations
- Text representations
- Similarity measures
- Retrieval techniques
- Classification or ranking techniques
- Other information available in the dataset

You are free to decide how these components should be used.

---



## Starting Point

You may use the approaches developed in Parts B and C as baselines.

A simple multimodal approach could conceptually look like:

```text
                 ┌── Text ──→ Text Representation ──┐
Product Listing ┤                                    ├──→ Matching
                 └── Image ─→ Image Representation ─┘
```

However, **this is only a conceptual example**.

You are encouraged to explore alternative architectures.

---



# Requirements



## 1. Establish a Baseline

Start with a reasonable baseline and report its performance.

Clearly describe:

- Architecture
- Features/representations
- Similarity method
- Matching strategy
- Hyperparameters

---



## 2. Improve Your System

Perform meaningful experiments to improve or better understand your system.

At least **two meaningful experiments** should be documented.

Your experiments should follow a clear structure:

```text
Hypothesis
    ↓
Experiment
    ↓
Result
    ↓
Analysis
    ↓
Conclusion
```

Do not simply change multiple components at once without analyzing their individual effects.

---



## 3. Multimodal Integration

Investigate how textual and visual information can be combined.

You may explore approaches such as:

- Score-level fusion
- Feature-level fusion
- Weighted combinations
- Learned combinations
- Multimodal embeddings
- Retrieval + reranking
- Ensemble approaches

These are examples only. You are free to explore other approaches.

---



## 4. Error Analysis

Analyze the final system's failures.

Identify examples involving:

- Similar products
- Different product variants
- Noisy titles
- Different images of the same product
- Visually similar products
- Missing information
- Conflicting visual and textual information

For a selection of errors, explain:

1. What did the model predict?
2. What was the correct answer?
3. Why did the model fail?
4. What could potentially fix the failure?

---



## 5. Ablation Study

Perform an ablation study to understand the contribution of different components.

For example:


| Configuration | Text | Image | Other | Score |
| ------------- | ---- | ----- | ----- | ----- |
| A             | ✓    | ✗     | ✗     | ...   |
| B             | ✗    | ✓     | ✗     | ...   |
| C             | ✓    | ✓     | ✗     | ...   |
| D             | ✓    | ✓     | ✓     | ...   |


Your experiments should help demonstrate **which components actually contribute to your final system**.

---



# Bonus Exploration

The following are optional directions you may investigate:

- Better embedding models
- Contrastive learning
- Metric learning
- Hard-negative mining
- Approximate nearest-neighbor search
- Clustering
- Retrieval and reranking
- Multilingual representations
- Data augmentation
- Ensemble methods
- Efficient inference
- Other approaches you discover

You are **not required to implement these techniques**.

A well-designed simple solution with strong experimental reasoning is preferable to an unnecessarily complicated system.

---



# Final Report

Your final submission should contain a concise report covering:

### 1. Approach

Explain your final architecture.

### 2. Experiments

Describe the experiments you performed and their results.

### 3. Analysis

Explain why certain approaches worked or failed.

### 4. Error Analysis

Discuss representative failure cases.

### 5. Ablation Study

Show the contribution of individual components.

### 6. Limitations

Discuss the limitations of your final approach.

### 7. Future Improvements

Describe what you would investigate if you had more time or resources.

---



# Submission Structure

Submit your work in the following format:

```text
Finale/
├── README.md
├── notebook.ipynb
├── src/
│   └── ...
├── results/
│   └── ...
└── report.pdf
```

The exact structure may be adapted depending on your implementation.

---



# External Resources

You are allowed to use:

- Kaggle notebooks
- Research papers
- GitHub repositories
- Documentation
- Pretrained models
- Open-source implementations
- AI-assisted tools

If you use an external implementation or substantially rely on an existing solution, clearly mention it in your report.

You should be able to explain **every major component of your submitted solution**.

---



# Evaluation

The final evaluation will consider:

- Final performance
- Understanding of the problem
- Quality of experimentation
- Experimental reasoning
- Multimodal understanding
- Error analysis
- Code quality
- Ability to justify design choices

The final metric is **only one part of the evaluation**.

Your solution will be discussed in detail during the interview.

---



## Final Thought

There is no single correct solution to this problem.

The objective of the Finale is to demonstrate that you can:

> **Understand → Experiment → Analyze → Improve → Explain**

Good luck! 🚀