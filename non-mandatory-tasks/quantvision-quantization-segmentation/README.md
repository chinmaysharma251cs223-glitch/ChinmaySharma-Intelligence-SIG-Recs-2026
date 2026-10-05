# QuantVision: Does Quantization Make Segmentation Unstable?

Welcome to the **QuantVision** recruitment challenge.

Your mission is to **train, evaluate, compare, and investigate** an instance-segmentation model under different numerical precisions.

You will compare:

* **FP32**
* **FP16**
* **INT8**

The goal is not simply to obtain the highest segmentation score.

Your goal is to understand:

> **Does quantization make an instance segmentation model less stable over time, and what evidence supports your conclusion?**

---

## Overview

In video, an instance segmentation model should produce reasonably consistent masks for the same object from one frame to the next.

However, changing a model from FP32 to FP16 or INT8 may alter its predictions.

This challenge asks you to investigate:

* Segmentation performance
* Temporal consistency
* Conditions under which FP32 and INT8 differ
* Possible causes of those differences
* Whether a proposed mitigation actually helps

The focus is on **experimentation, reasoning, and evidence**, rather than simply training a model.

A well supported negative result is completely valid.

---

## Dataset

Use the provided **Customized KITTI-MOTS Dataset**.

The link for the dataset: https://drive.google.com/file/d/19YAN104umEYFaOpE854Yi7UI5mHL5D6j/view?usp=sharing

The dataset is derived from KITTI-MOTS and has been curated specifically for this challenge.

The primary class for this task is:

* **Car**

The dataset contains instance-level segmentation annotations from driving sequences and is divided into:

* **Training data**
* **INT8 calibration data**
* **Evaluation data**

The splits are **sequence-disjoint**.

The provided dataset contains approximately:

* **2,250 training images (161 with masks; the rest are unlabeled context - see `has_label` below)**
* **120 calibration images**
* **150 evaluation frames**

The evaluation data contains consecutive frames from held-out sequences for temporal analysis. Ground truth within clips is sparse (see `has_label` in `manifest.csv`): measure stability on the consecutive frames, reference ground-truth motion where labels exist, and state your limits.

The training data may also be used to investigate the effect of increasing the amount of training data.

### Package Contents & Annotation Format

* `train/images/` - 2,250 images named `<sequence>_<frame>.png`.
* `train/masks/` - for the 161 labeled images: `<base>_inst.png` (0 = background, 1..N = car instances), `<base>_ignore.png` (1 = ignore region - always exclude from metrics), `<base>_meta.json` (instance-ID mapping, mask areas, source reference).
* `train_subsets.json` - nested `train_500 ⊂ train_1250` file lists (subsets of the 2,250).
* `calibration/images/` - 120 images, no masks (INT8 PTQ calibration only).
* `evaluation/images/` - 150 images: 4×30 consecutive clips (`clip_id` A-D, `clip_position` 0-29)
  + 30 diverse frames. No masks shipped - scoring uses held-back ground truth.
* `manifest.csv` - one row per file: split, sequence, frame, paths, dimensions, object counts, `clip_id`/`clip_position`, and `has_label` (1 = ground truth exists and is used in scoring; 0 = unlabeled context - use freely, never treat as empty).

### Important Rules

* Do not use evaluation data for INT8 calibration.
* Use the same evaluation data for FP32, FP16 and INT8.
* Do not introduce sequence leakage.
* Do not randomly distribute frames from the same sequence across different splits.
* Only treat frames as consecutive when they are actually consecutive within the same sequence.
* Instance IDs may be used to associate the same object across consecutive frames for temporal analysis.
* A separate tracking system is not required.
* The primary scored class is car.

---

# Task Stages

## Stage 1 - Build & Establish the Baseline

### Objective

Build a reliable instance-segmentation pipeline and establish how FP32, FP16 and INT8 behave under the same evaluation conditions.

### 1.1 Model Training

Train or fine-tune a suitable **lightweight instance-segmentation model** using the provided training data.

The main training set contains **2,250 images, of which 161 carry instance masks** (the rest are unlabeled video context, flagged by `has_label=0`).

### 1.2 Data Scaling Experiment

Investigate how the amount of training data affects model behaviour using progressively larger subsets such as:

* 500 frames
* 1,250 frames
* 2,250 frames

Keep the evaluation set fixed.

Keep the calibration set fixed.

Keep the model family and training methodology reasonably consistent.

The purpose is to investigate the trend, not to maximize the number of training runs.

### 1.3 Precision Variants

Create:

* FP32
* FP16
* INT8

versions of the same trained model.

For INT8, use the provided calibration dataset.

Do not use evaluation data during calibration.

### 1.4 Segmentation Performance

Evaluate all three models on exactly the same evaluation data.

Use appropriate segmentation metrics such as:

* Mask mIoU
* Dice / F1
* Precision
* Recall

You do not need to report every possible metric.

Choose metrics appropriate to your implementation and explain your choice.

### 1.5 Temporal Behaviour

The evaluation set contains consecutive frames from held-out driving sequences.

Investigate how segmentation predictions change from one frame to the next.

You may investigate:

* Mask overlap
* Mask-area changes
* Boundary changes
* Confidence changes
* Objects appearing or disappearing
* Object-level consistency

Choose **one main metric for temporal stability** and justify your choice.

Your explanation should include:

* What the metric measures
* Why it is appropriate
* What higher/lower values indicate
* Important limitations

Be careful when interpreting frame-to-frame changes.

A car can legitimately move between frames.

Therefore:

> **A changing segmentation mask is not automatically evidence of model instability.**

### Deliverables

Submit:

* Training configuration
* FP32 / FP16 / INT8 results
* Segmentation metrics
* Temporal-stability metric and results
* Data-scaling results, if performed
* Training/evaluation plots
* Representative visualizations
* Saved predictions
* Short explanation of the evaluation methodology

---

## Stage 2 - Investigate & Explain

### Objective

Go beyond the headline numbers and investigate **where and why** FP32 and INT8 behave differently.

Do not stop at:

> "INT8 has lower mIoU."

Investigate the conditions under which the difference occurs.

You may consider factors such as:

* Small vs large objects
* Object density
* Occlusion
* Confidence changes
* Mask-boundary changes
* Different sequences
* Different scene characteristics
* Post-processing
* Quantization effects
* Calibration behaviour

These are examples only.

Your conclusion should come from your experiments.

### 2.1 Form a Hypothesis

Identify an observation from Stage 1 and formulate a testable hypothesis.

For example:

> "INT8 degradation is concentrated on small objects."

or:

> "The observed temporal instability is mainly caused by post-processing."

Your hypothesis should be motivated by evidence.

### 2.2 Test the Hypothesis

Design an experiment that can support or reject your hypothesis.

Clearly describe:

* What you observed
* Your hypothesis
* Experimental setup
* Variables
* Metrics
* Results
* Interpretation

### 2.3 Consider an Alternative Explanation

Consider at least one other plausible explanation for your observation.

For example, if one evaluation sequence performs poorly under INT8, investigate whether the effect may actually be related to:

* object scale
* scene density
* occlusion
* motion
* another property of the data

Your analysis should attempt to distinguish between plausible explanations.

### 2.4 Temporal Reasoning

When analyzing temporal stability, consider:

**Ground-truth temporal change**

versus

**FP32 prediction change**

versus

**INT8 prediction change**

A correct model can change its prediction because the scene itself changes.

Your analysis should explain how you distinguish **real object motion** from changes introduced by the model.

### Deliverables

Submit:

* Main hypothesis
* Observation motivating the hypothesis
* Experimental methodology
* Alternative explanation considered
* Results
* Supporting plots/visualizations
* Final interpretation

---

## Bonus Stage - Mitigate & Conclude Bonus

### Objective

Use what you learned in Stage 2 to design **one method to reduce INT8 degradation or temporal instability**.

Your method should be motivated by your diagnosis.

Possible approaches include:

* Improved calibration
* Different quantization configuration
* Keeping selected components at higher precision
* Quantization-aware training
* Training or preprocessing changes
* Another technically justified approach

These are examples only.

You are not required to use one of them.

### 3.1 Before vs After

Compare:

**INT8**

vs.

**INT8 + your method**

Evaluate both using:

* Segmentation performance
* Temporal stability

### 3.2 Analyze the Trade-off

Do not evaluate the method using only one metric.

For example:

| Model         | Segmentation | Temporal Stability |
| ------------- | -----------: | -----------------: |
| INT8          |         0.78 |               0.18 |
| INT8 + method |         0.76 |               0.09 |

An improvement in temporal stability accompanied by a reduction in segmentation quality is a **trade-off**, not automatically a success.

Discuss whether the mitigation is actually useful.

### 3.3 Final Conclusion

Your final report should answer:

> **Does quantization make your instance-segmentation model less stable over time?**

Support your conclusion using:

* Segmentation results
* Temporal-stability results
* Relevant condition-wise analysis
* Hypothesis testing
* Alternative explanations
* Mitigation results

There is no predetermined answer.

### Deliverables

Submit:

* Mitigation method
* Motivation for selecting it
* Before/after results
* Segmentation comparison
* Temporal comparison
* Trade-off analysis
* Final conclusion
* Limitations

---

# What to Submit

Submit:

* Source code
* Model training code
* Model conversion code
* INT8 calibration code
* Evaluation code
* Saved predictions
* Results
* Plots
* Representative examples
* Experiment log
* Final report

Your experiment log should record the important experiments you tried and what you learned from them.

You should be able to explain how you obtained your important results.

Your final report should clearly distinguish between:

**Observation → Hypothesis → Experiment → Result → Conclusion**

---

# Important Experimental Requirements

## Sequence-Level Splitting

Training, calibration and evaluation must be **sequence-disjoint**.

Do not randomly split individual frames from the same sequence across different splits.

Explain your splitting strategy.

## Calibration

Do not use evaluation data for INT8 calibration.

Clearly state which data was used for calibration.

## Fixed Evaluation

Use exactly the same evaluation data for:

* FP32
* FP16
* INT8

Do not change the evaluation set between models.

## Temporal Evaluation

Only treat frames as consecutive when their frame numbers are actually consecutive within the same sequence.

Do not assume that nearby samples are consecutive.

## Instance Identity

Instance IDs may be used to associate the same car across consecutive frames for temporal segmentation analysis.

A separate tracking system is not required.

---

# Three Things to Watch Carefully

### 1. Sequence Leakage

Frames from the same driving sequence are related.

Randomly splitting frames can cause information leakage between training and evaluation.

Use a sequence-aware split.

### 2. Motion vs Instability

A prediction can change because the object moved.

Do not automatically interpret frame-to-frame mask changes as model instability.

Consider the underlying ground-truth motion when interpreting temporal behaviour.

### 3. Stability vs Accuracy

A mitigation that improves temporal stability may reduce segmentation quality.

Evaluate both before deciding whether the method is useful.

---

# Evaluation Criteria

This challenge is **not mainly about achieving the highest segmentation score**.

We care about:

* Experimental design
* Understanding of instance segmentation
* Understanding of quantization
* Temporal reasoning
* Hypothesis formation
* Quality of experiments
* Evidence-based conclusions
* Honest interpretation of results
* Reproducibility

A result showing **no meaningful difference** between FP32 and INT8 is completely valid if your experiment supports that conclusion.

A result showing significant INT8 instability is also valid if supported by evidence.

There is no predetermined conclusion.

---

# Key Focus Areas

## Experimental Design

Design experiments that can distinguish between plausible explanations.

Avoid making causal claims from a single aggregate number.

## Temporal Reasoning

Distinguish between:

* Real changes in the scene
* Changes caused by the model
* Changes caused by post-processing

## Quantization Analysis

Investigate why FP32 and INT8 differ rather than simply reporting the difference.

## Mitigation Quality

Evaluate your mitigation using both:

* Segmentation quality
* Temporal stability

Discuss trade-offs rather than optimizing only one metric.

## Reproducibility

Important experiments should be reproducible from the submitted code and configuration.

---

# Expectations

You are expected to:

* Document important experiments
* Keep the code organized
* Record model and training settings
* Record INT8 calibration settings
* Explain how metrics were calculated
* Keep the evaluation protocol consistent
* Explain experimental decisions
* Support conclusions with evidence

A suggested project structure is:

```text
quantvision/
│
├── src/
│   ├── train/
│   ├── quantization/
│   ├── evaluation/
│   └── analysis/
│
├── configs/
├── experiments/
├── predictions/
├── plots/
├── logs/
└── report/
```

You do not have to follow this exact structure, but your submission should remain easy to understand and reproduce.

---

# Compute Expectations

The task is designed to be feasible on a standard cloud GPU such as:

* Kaggle T4
* Kaggle P100
* Google Colab T4
* Google Colab L4

You are not expected to use:

* Multi-GPU systems
* High-end server hardware
* Very large segmentation architectures

The dataset has been sized so that training is meaningful without making computation the primary challenge.

The main training workload is **2,250 images with 161 labeled frames** (a YOLOv8n-seg-class model fine-tunes in minutes on a T4), with smaller subsets available for the optional data-scaling experiment.

You should spend more time **understanding and analyzing your results** than waiting for training.

---

# Tips for Success

* Do not focus only on the final mIoU.
* Look at where the models disagree.
* Visualize representative examples.
* Think carefully about object motion.
* Question your initial hypothesis.
* Consider alternative explanations.
* Record important unsuccessful experiments.
* Explain why your chosen metric is appropriate.
* Support conclusions with evidence.

---
