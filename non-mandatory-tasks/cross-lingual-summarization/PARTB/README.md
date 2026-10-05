# Part B: Building the Cross-Lingual Summarizer

## Goal

Build a model/pipeline that takes a document in the source language and produces a summary in the target language. This is document understanding plus cross-lingual generation, not sentence-level translation, the output should read as a genuine summary, not a translated version of one or two sentences pulled from the article.

## What You Need to Do

1. Build a summarizer using whichever approach you prefer: a pretrained multilingual summarization/generation model you fine-tune on the CrossSum training subset, a pipeline that summarizes in the source language first and then translates the summary, or an end-to-end approach that goes straight from source document to target-language summary. All are valid, but justify your choice, and note that a "summarize then translate" pipeline is architecturally different from true cross-lingual generation, discuss that tradeoff if you go that route.
2. Generate summaries for the CrossSum test subset.
3. Compare your generated summaries against the human-written reference summaries. Standard n-gram metrics like ROUGE don't work well across languages, so use a semantic similarity metric instead (e.g. multilingual embedding similarity between your generated summary and the reference, using your Part A embedding model).

## Encouraged Experimentation

- Compare "summarize then translate" against a more direct cross-lingual approach and discuss quality differences
- Try conditioning generation length/style and see how it affects semantic similarity to the reference
- Look at a handful of outputs by hand (especially if you or a teammate can read both languages) and note qualitative issues you see beyond what the automatic metric captures

## Deliverables

- `/PartB/src/` with your summarizer implementation
- `/PartB/output/generated_summaries.csv` with (doc_id, generated_summary, reference_summary, similarity_score) for the test subset
- `/PartB/README.md` documenting your architecture/approach, why you chose it, your evaluation results, and at least 3 example outputs with brief commentary