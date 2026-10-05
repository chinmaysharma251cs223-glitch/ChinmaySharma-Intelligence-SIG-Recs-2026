# Finale: Handling Long Documents

## Task Overview

This is the culmination of Parts A, B, and C. You've built embeddings, a cross-lingual summarizer, and a faithfulness evaluator. Most summarization models (including whatever you built in Part B) have a limited context window, they can't take an arbitrarily long document in a single pass. The Finale asks you to design and implement a solution for documents that exceed that limit.

## Part 1: Design

Write a short design document (1-2 pages) covering:

- How would you summarize a document too long for your Part B model to process in one pass, while still producing a single coherent target-language summary (not just a list of disconnected chunk summaries)?
- How do you keep the summary faithful across this process? Chunking and summarizing independently risks losing cross-chunk context (a claim in chunk 1 might be contradicted or clarified in chunk 3), how does your design account for that?
- How would you use your Part C faithfulness evaluator here, does it still work directly, or does it need adapting for long documents?
- What you'd change or add with more time/resources.

## Part 2: Implementation

Implement your long-document solution. At minimum:

1. It should handle documents at least 3-4x longer than your Part B model's normal input limit.
2. Run it on a set of long documents from the CrossSum test subset (or, if none are long enough, concatenate a few related short articles to simulate a long document, note if you do this).
3. Run your Part C faithfulness evaluator (or your adapted version of it) on the output.

## Deliverables

- `/Finale/design.md` with your design document
- `/Finale/src/` with your implementation
- `/Finale/output/long_doc_summaries.csv` with generated summaries and faithfulness verdicts for your long-document test set
- `/Finale/README.md` summarizing what you built, how it compares to naively truncating the document or chaining independent chunk summaries, what worked, what didn't, and what you'd do differently with more time

## Evaluation

We're looking at:

- Whether your long-document approach genuinely addresses cross-chunk context, not just chunking and summarizing independently
- Faithfulness of the final summary, as measured by your Part C evaluator
- Quality of reasoning in your design document, especially around the tradeoffs of different long-document strategies
- How well you've documented limitations

Novel ideation and smart experimentation are valued over a "safe" approach that technically works but doesn't show much thought.