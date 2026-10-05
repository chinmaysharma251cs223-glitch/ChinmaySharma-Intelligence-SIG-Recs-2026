# Cross-Lingual Summarization Track

## Task Overview

This task walks you through building a system that reads a document in english language and produces a faithful summary in a spanish language. This is not sentence-level machine translation, it's document understanding combined with cross-lingual generation: you need to actually understand what the source document says, then express that understanding as a summary in another language, without introducing information that isn't in the source (hallucination).

## Dataset

This task uses **CrossSum** (Hasan et al., ACL 2023), a large-scale cross-lingual summarization dataset with 1.7 million article-summary pairs across 1500+ language pairs and 45 languages, built from BBC news content. Each example has an article in a source language and a human-written summary in a different target language.

- Paper: https://aclanthology.org/2023.acl-long.143/
- Dataset (Hugging Face): https://huggingface.co/datasets/csebuetnlp/CrossSum
- Load example:
```python
from datasets import load_dataset
ds = load_dataset("csebuetnlp/CrossSum", "spanish-english") 
```
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (non-commercial research use, fine for this task)


## Structure

| Part | Folder | What you'll build |
|------|--------|--------------------|
| A | `/PartA` | Embeddings over source-language documents |
| B | `/PartB` | A cross-lingual summarizer (document in language X, summary in language Y) |
| C | `/PartC` | A faithfulness evaluator that checks summaries against the source for hallucination |
| Finale | `/Finale` | Your own design for handling long documents, implemented and tested |

Each folder has its own README with the exact goal, deliverables, and submission requirements for that part. Read each one before starting that part.

## General Rules

- Novel ideation and smart experimentation are valued over raw performance. If you try something unconventional and it doesn't beat the baseline, explain your reasoning, it still counts in your favor.
- Every part needs its own README documenting your approach, choices, and results. Code without a writeup will not be evaluated fully.
- Commit your work incrementally. We look at commit history, not just the final state.
