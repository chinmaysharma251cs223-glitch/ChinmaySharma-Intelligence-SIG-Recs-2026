# Learning to Parallelize Code

## Overview

This challenge explores how **machine learning can be applied to automatic code parallelization**.

You will work through two progressively harder problems:

1. Extract useful supervision from a raw C/OpenMP code corpus and build an ML model for loop parallelization and OpenMP clause prediction.
2. Use ML predictions to guide an LLM when generating an OpenMP directive.

You are free to choose your own representations, models, and experiments. The goal is to demonstrate good ML reasoning rather than reproduce one particular solution.

---

# Dataset

The dataset is a raw corpus of C programs containing both OpenMP-annotated and unannotated code. It contains approximately **500 C files**, with roughly half of the loops in the corpus are OpenMP-annotated. The corpus contains multiple loops per file in many cases, and the OpenMP directives include clauses such as `private`, `reduction`, `firstprivate`, and `lastprivate`, including combinations of these clauses.

The dataset is intentionally provided as source code rather than as a ready-made machine-learning table. Part of the challenge is to inspect the corpus, determine how OpenMP directives correspond to the loops they control, and construct an appropriate set of labels or other supervision for your experiments. You should also decide how to handle examples that are ambiguous, unusual, or difficult to process, and explain those decisions.

---

# Problem 1 - Loop Parallelization and OpenMP Pattern Prediction

Build an ML-based system that analyzes loops and predicts their parallelization properties.

At minimum, consider:

```text
PARALLEL
NON_PARALLEL
```

For loops identified as parallel, also predict the OpenMP parallelization pattern or clause properties. Possible categories include:

```text
DO-ALL
PRIVATE
REDUCTION
PRIVATE + REDUCTION
```

The dataset also contains other OpenMP clauses, including `firstprivate` and `lastprivate`. You may incorporate these into your formulation if appropriate.

You are not required to use the categories above exactly. You should determine a sensible formulation based on your exploration of the dataset and justify your choices. For example, clause properties may be treated as multi-label predictions rather than as a single multiclass target.

### Requirements

- Explore the raw dataset and explain how you extracted or constructed your labels.
- Build at least one baseline.
- Evaluate on held-out data.
- Report suitable metrics.
- Analyze failure cases.
- Explain your representation and model choices.
- Compare at least two approaches or representations.

You can use source tokens, ASTs, handcrafted features, embeddings, graphs, intermediate representations, or any other representation you can justify.

For classification results, report appropriate per-class or per-label metrics rather than relying only on overall accuracy.

---

# Problem 2 - OpenMP Directive Generation

Use the analysis from Problem 1 to generate an appropriate OpenMP directive for a loop.

You may use **rule-based methods, static analysis, retrieval, machine learning, language models, or combinations of these**. The goal is to translate your analysis of the loop into a valid and appropriate OpenMP directive.

At minimum, compare your approach against a reasonable baseline.

### Requirements

* Generate an OpenMP directive for the same unseen examples used for evaluation in Problem 1.
* Ensure that generated directives are syntactically valid OpenMP where applicable.
* Compare generated directives against the reference directives and their relevant clause properties.
* Report appropriate metrics.
* Analyze cases where your approach succeeds or fails.
* Explain how information from Problem 1 is used during generation.

Consider metrics such as:

- exact directive match
- normalized directive/clause match
- parallel vs non-parallel correctness
- private/reduction/other clause accuracy
- valid OpenMP syntax

You may design additional metrics if you can justify them.

---

# What We Care About

We are interested in:

- problem formulation
- dataset exploration and label construction
- representation choices
- sensible baselines
- experimentation
- failure analysis
- clear reasoning

A simple model with strong analysis is better than a complicated model with no justification.

There is no single correct representation, model, or solution. We are interested in the reasoning behind your choices and what you learn from the experiments.

---

# Resources

You may use any relevant documentation, libraries, papers, or other resources. Some useful starting points include:

- OpenMP Specification: https://www.openmp.org/specifications/
- LLVM documentation: https://llvm.org/docs/
- NetworkX documentation: https://networkx.org/documentation/stable/
- PyTorch Geometric documentation: https://pytorch-geometric.readthedocs.io/

These are starting points rather than required tools or approaches. You are free to use other resources and libraries where appropriate.

# Submission

Submit:

```text
README.md
src/
    ...
experiments/
    ...
results/
    ...
```

Your README should briefly describe:

1. Dataset exploration and label construction
2. Approach
3. Representation
4. Model(s)
5. Experimental setup
6. Results
7. Failure cases
8. What you would try next

You may use existing libraries, pretrained models, and LLM APIs where appropriate. Clearly state what you used.

**There is no single correct solution.**
