# Part C: Evaluating Faithfulness

## Goal

Build an evaluator that checks whether a generated summary (in the target language) actually reflects what's in the source document (in the source language), without hallucinating information that isn't there. This is a harder question than "does it read well," a fluent summary can still invent facts, and that's exactly what this evaluator needs to catch.

## Approaches

You need a way to compare content across two languages. A couple of valid approaches, pick one or combine them:

1. **Back-translation + monolingual faithfulness check:** translate the generated summary back into the source language, then check whether each claim in the back-translated summary is supported by the source document. This can be done with a monolingual entailment/NLI model (does the source document entail each summary sentence) or a simpler overlap-based heuristic as a baseline.
2. **Direct multilingual entailment:** use a multilingual NLI model that can take a premise in one language and a hypothesis in another (some multilingual NLI models support this reasonably well, others don't, test this on a few known-correct pairs first before trusting it).

Either way, your output should be sentence-level or claim-level: for each sentence/claim in the summary, a verdict of supported, unsupported, or contradicted, with the source passage it was checked against.

## What You Need to Do

1. Implement your chosen faithfulness-checking approach in `/PartC/src/`.
2. Run it on the summaries your Part B model generated for the test subset.
3. Manually annotate faithfulness (by hand) for a small sample (~20-30 summaries) as a sanity check, then compare your automatic evaluator's verdicts against your manual judgments to see how reliable it is.

## Encouraged Experimentation

- Try both approaches above (if you have language ability to check outputs) and compare which catches more real hallucinations
- Look for patterns: does your Part B summarizer hallucinate more on longer documents, more on certain topics, or specific types of claims (numbers, names, dates)

## Deliverables

- `/PartC/src/` with your faithfulness evaluator
- `/PartC/output/faithfulness_report.csv` with per-summary (or per-sentence) verdicts for the test subset
- `/PartC/output/manual_annotation_sample.csv` with your manual annotations and how they compared to the automatic evaluator
- `/PartC/README.md` documenting your approach, its limitations, how it compared to manual annotation, and what kinds of hallucinations (if any) you found in your Part B model's output

## Notes

Be honest about your evaluator's own limitations, an imperfect but well-understood evaluator is more valuable here than one that looks clean but you can't explain.