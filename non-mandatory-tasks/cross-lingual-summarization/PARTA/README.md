# Part A: Building Embeddings over Source-Language Documents

## Goal

The goal of this part is to experiment with representing source-language documents from CrossSum as embeddings. These will be useful context for Part C (checking whether a summary is faithful to its source).


## What You Need to Do

1. Chunk each source document into reasonable units (sentence-level, paragraph-level, or a fixed token window, your choice, but justify it).
2. Generate embeddings for each chunk using a multilingual embedding model (e.g. a multilingual sentence transformer) or one you train/fine-tune yourself. It needs to be multilingual since you'll compare it against target-language text in Part C.
3. Save your embeddings as a CSV with (doc_id, chunk_id, chunk_text, language, embedding) columns and upload it to `/PartA/output/`.

## Encouraged Experimentation

- Compare a general-purpose multilingual embedding model against one fine-tuned for cross-lingual similarity specifically
- Check whether semantically similar sentences in different languages actually land close together in your embedding space (a quick sanity check: embed a source sentence and its reference summary, see how close they are)
- Try different chunking strategies and note tradeoffs

Novel ideation and smart experimentation are highly appreciated, more so than squeezing out marginal performance gains.

## Deliverables

- `/PartA/output/embeddings.csv` with (doc_id, chunk_id, chunk_text, language, embedding) pairs
- `/PartA/README.md` documenting:
  - Your chunking strategy and why you chose it
  - Your embedding model choice and why, including your cross-lingual sanity check
  - Any experiments you ran