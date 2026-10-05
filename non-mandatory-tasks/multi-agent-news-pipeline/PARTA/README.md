# Part A: Building Embeddings on the NewsQA Corpus

## Goal

The goal of this part is to experiment with representing news articles as embeddings, using the fixed NewsQA subset provided in this folder. This corpus is used throughout the rest of the task (Part B's Fact-Checker Agent, Part C's pipeline, and the Finale), so the embeddings you build here should be ones you can reuse or rebuild consistently.

## Dataset

You can get the dataset using

from datasets import load_dataset dataset = load_dataset("lucadiliello/newsqa")

Do not substitute your own dataset here.


## What You Need to Do

1. Chunk the articles into reasonable units (sentence-level, paragraph-level, or a fixed token window, your choice, but justify it).
2. Generate embeddings for each chunk using an embedding approach of your choice (pretrained sentence transformers, a model you train yourself, or anything in between).
3. Save your embeddings as a CSV with (article_id, chunk_id, chunk_text, embedding) columns and upload it to `/PartA/output/`.

## Encouraged Experimentation

Different forms of experimentation are encouraged, irrespective of how "good" the resulting embeddings look on paper. Some ideas:

- Compare a pretrained embedding model against one you fine-tune or train from scratch
- Try different chunking strategies and see how much it affects downstream retrieval quality (you'll test this properly in Part C)
- Try dimensionality reduction and visualize the embedding space (e.g. clustering articles by topic)

Novel ideation and smart experimentation are  appreciated, more so than squeezing out marginal performance gains.

## Deliverables

- `/PartA/output/embeddings.csv` with (article_id, chunk_id, chunk_text, embedding) pairs
- `/PartA/README.md` (this can be a copy of this file with a "My Approach" section added) documenting:
  - Your chunking strategy and why you chose it
  - Your embedding approach and why you chose it
  - Any experiments you ran and what you observed

## Notes

Keep your code in `/PartA/src/` and make sure it can be re-run end to end (raw NewsQA subset in, CSV out) without manual steps.