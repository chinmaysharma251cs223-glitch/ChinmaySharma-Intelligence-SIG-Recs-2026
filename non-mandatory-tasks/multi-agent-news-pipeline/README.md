# Simple Multi-Agent Pipeline

## Task Overview

This task walks you through building a small, modular multi-agent NLP system for news articles, from the ground up. You'll start by representing a fixed news corpus as embeddings, then build three single-purpose agents, then chain those agents into a working pipeline with a controller, and finally apply the full system in the Finale.

The goal isn't to build the most complex agent framework possible. It's to understand, at a fundamental level, how agents are structured, how they communicate, and how a controller decides what happens next. Simple, well-reasoned pipelines are valued over unnecessarily complicated ones.

## Dataset

All parts use the **NewsQA dataset** (CNN news articles with human-written questions and answers). A fixed subset is provided in `/PartA/data/`. Using this shared corpus across all parts keeps submissions comparable and lets you reuse work between parts.

## Structure

| Part | Folder | What you'll build |
|------|--------|--------------------|
| A | `/PartA` | Embeddings on the provided NewsQA corpus |
| B | `/PartB` | Three single-purpose agents: Summarizer, Query-Answering, Fact-Checker |
| C | `/PartC` | A controller that chains the three agents into a pipeline |
| Finale | `/Finale` | The full system applied end to end, with a design writeup |

Each folder has its own README with the exact goal, deliverables, and submission requirements for that part. Read each one before starting that part.

## General Rules

- Use the NewsQA subset provided in `/PartA/data` for Part A. Do not substitute a different dataset for this part, the goal is a shared baseline across all recruits.
- From Part B onward, you're free to bring in additional tools or techniques as long as you document what you used and why. The underlying dataset stays NewsQA throughout.
- Novel ideation and smart experimentation are valued over raw performance. If you try something unconventional and it doesn't work as well, explain your reasoning in the README, it still counts in your favor.
- Every part needs its own README documenting your approach, choices, and results. Code without a writeup will not be evaluated fully.
- Commit your work incrementally. We look at commit history, not just the final state.

## Submission

Fork/clone this repository, complete each part in its respective folder, and submit your final repository link by the deadline communicated separately.
