# Part B: Building the Agents

## Goal

Build three independent, single-purpose agents, all operating on NewsQA news articles. Each agent should do exactly one job and do it well. At this stage, the agents are standalone, they do not call each other yet (that happens in Part C).

You may implement each agent using an LLM call with a well-designed prompt, a smaller trained/fine-tuned model, a rule-based system, or any combination, as long as you can justify the choice in your README.

## The Three Agents

### 1. Summarizer Agent
- **Input:** a NewsQA article
- **Output:** a concise summary of the article
- **What we're looking at:** does the summary preserve the key information without just copying large spans of the original text

### 2. Query-Answering Agent
- **Input:** a user's natural language question about an article, plus the article/corpus as context
- **Output:** an answer grounded in the article/corpus, with the specific passage(s) it drew from. NewsQA's own question-answer pairs are a good source of test questions.
- **What we're looking at:** does the agent retrieve and ground its answer in the actual text, rather than just guessing

### 3. Fact-Checker Agent
- **Input:** a claim/statement extracted from an article, plus access to your Part A corpus and embeddings
- **Output:** a verdict (corroborated / contradicted / unverifiable) along with the specific chunk(s) from the corpus used as evidence
- **What we're looking at:** does the agent actually retrieve and cite relevant evidence rather than just guessing

## What You Need to Do

1. Implement each agent as a standalone, callable unit (a function, class, or script, your choice) in `/PartB/src/`.
2. For each agent, write at least 3 example runs (input and output) demonstrating it works, saved in `/PartB/examples/`.
3. Document each agent's design in your README: what approach you used, what prompt/model/logic it relies on, and its known limitations.

## Encouraged Experimentation

- Try more than one approach for a given agent (e.g. two different retrieval strategies for the Fact-Checker) and compare
- Add simple guardrails, e.g. what does the Fact-Checker Agent do when there's genuinely no relevant evidence in the corpus
- Use NewsQA's real question-answer pairs to evaluate the Query-Answering Agent against ground truth

## Deliverables

- `/PartB/src/` with one implementation per agent (clearly named/organized)
- `/PartB/examples/` with example input/output runs for each agent
- `/PartB/README.md` documenting each agent's design, approach, and limitations

## Notes

Keep the three agents independent at this stage, no agent should call another yet. That comes in Part C.