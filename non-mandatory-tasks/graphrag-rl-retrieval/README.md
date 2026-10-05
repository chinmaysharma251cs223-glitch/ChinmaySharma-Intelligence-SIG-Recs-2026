# GraphRAG - From Heuristic Retrieval to RL-Trained Retrieval

## Overview

Your task is to explore how **GraphRAG** - retrieval over a knowledge graph instead of flat vector search - performs against plain vector RAG, and then go one step further: explore how **RL-trained retrieval** (the newest branch of the same idea) performs against GraphRAG's fixed heuristics.

The goal: understand where graph structure actually helps retrieval (and where it doesn't), and whether teaching the retrieval process with reinforcement learning meaningfully beats hand-designed heuristics.

## Important Guidelines

- **Packaged end-to-end libraries are not allowed for the core experiments** (e.g. calling `graphrag.index()` / `graphrag.query()` as a black box). You should be building and orchestrating the pipeline stages yourself - entity/relationship extraction, graph construction, community detection/summarization, and graph-based retrieval - even if you use lower-level libraries for individual pieces.
- You may use **low-level building blocks**, such as:
  - An LLM API (or a local model) for entity/relationship extraction from your documents
  - `networkx` (or similar) for graph structure and community detection algorithms (e.g. Leiden, Louvain)
  - A vector DB (FAISS/Chroma) for any embedding-based lookups your retrieval logic needs
- For reference on what the full pipeline should look like, read the [GraphRAG paper](https://arxiv.org/abs/2404.16130) and its [implementation](https://github.com/microsoft/graphrag) - use it to understand each stage, not to import and run.
- For the RL-trained retrieval side, use any of:
  - [Graph-R1](https://github.com/LHRLAB/Graph-R1) (ICML 2026)
  - [GraphRAG-R1](https://github.com/ycygit/GraphRAG-R1) (WWW 2026)
  - Training an RL policy from scratch is genuinely out of reach in the timeframe - using their released checkpoint for this part is expected, not a shortcut.
- Use any **open document set or benchmark**, from:
  - Microsoft's own [GraphRAG tutorial dataset](https://github.com/microsoft/graphrag/blob/main/docs/get_started.md)
  - A small cluster of related Wikipedia articles or news coverage of one event
  - Standard multi-hop QA benchmarks: HotpotQA, MuSiQue, 2WikiMultiHopQA
- If you are feeling super enthusiastic, you are free to also implement your own retrieval policy or train a lightweight RL agent yourself instead of using a released checkpoint. This will definitely carry bonus points!

**NOTE:** You are not limited by any of the above mentioned tools and datasets as long as it fits the guidelines.

Explain your approach clearly, and include references to any papers or repos you use. Creativity, depth of analysis, and clear visualization will be rewarded.

## Requirements

### Input
- A document set or corpus (for GraphRAG/plain RAG), or a multi-hop QA benchmark subset (for the RL-trained variants).
- A fixed set of test questions, ideally spanning both simple lookup and multi-hop/synthesis question types.

### Output
- Retrieved context + generated answer for each system, for each test question.
- A comparison table across systems on your chosen metrics.

## Experiments

### Exp 1: GraphRAG Baseline Build a GraphRAG-style pipeline on your chosen document set - entity/relationship extraction, graph construction, community detection/summarization, and graph-based retrieval - orchestrated yourself using the low-level building blocks listed above, not a packaged library call. Evaluate it on your test questions, paying attention to how it performs differently on simple lookup vs multi-hop/synthesis questions.

### Exp 2: GraphRAG vs RL-Trained Retrieval Set up and run one of Graph-R1 or GraphRAG-R1 (using their released checkpoint) on the same or a comparable question set. Compare its retrieval decisions and answer quality against GraphRAG's fixed heuristics - does learned retrieval actually beat hand-designed rules, as the papers claim?

## Bonus Experiments

### Exp 3: Train Your Own Lightweight RL Retrieval Policy Instead of using a released checkpoint for Exp 2, implement and train a small-scale RL retrieval policy yourself (e.g. a simplified retrieve/stop decision with a lightweight reward) on top of your GraphRAG pipeline. This is a genuinely heavier lift on top of an already from-scratch pipeline - real training risk (reward design, instability) - so treat it as a stretch, not a requirement.

### (Optional) Exp 4: Visualization & Interactive Demo Build a small Streamlit app where a question can be run through all your systems side by side, showing the retrieved context and (for the RL variant) its reasoning trace. Optionally visualize the entity subgraph each system walked to answer a question (e.g. via `streamlit-agraph` or `pyvis`). Discuss what "better retrieval" actually looks like across your systems.

## 🎯 Submission

Submit:
- A clean Colab / Jupyter notebook or GitHub repo with your code and results.
- A short report (2-3 pages) summarizing:
  - Approach & dataset(s)
  - Experimental results
  - Insights and takeaways

Good luck!