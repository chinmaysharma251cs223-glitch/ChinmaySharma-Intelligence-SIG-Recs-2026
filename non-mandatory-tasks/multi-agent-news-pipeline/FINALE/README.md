# Finale: News Article Multi-Agent System

## Task Overview

This is the culmination of Parts A, B, and C. You've built embeddings, standalone agents, and a controller that chains them. In the Finale, everyone builds the same system, a multi-agent assistant for news articles, using the corpus from Part A (or an extended version of it, see below).

We're fixing the problem this time so submissions are comparable across recruits, unlike an open-ended design, everyone is solving the same task, which means depth of execution and quality of reasoning are what set submissions apart.

## The System

Build a multi-agent system with the following three agents:

### 1. Summarizer Agent
- **Input:** a news article
- **Output:** a concise summary of the article

### 2. Query-Answering Agent
- **Input:** a user's natural language question, plus the article/corpus as context
- **Output:** an answer grounded in the article/corpus, with the specific passage(s) it drew from

### 3. Fact-Checker Agent
- **Input:** a claim extracted from an article
- **Output:** a verdict, corroborated, contradicted, or unverifiable, based on cross-referencing the rest of the corpus, along with the supporting/contradicting passage(s)

## Part 1: Design

Before building, write a short design document (1-2 pages) covering:

- How the controller routes a raw input (e.g. "here's an article and a question about it") to the right agent(s)
- How the Fact-Checker Agent handles claims that touch multiple articles, or where evidence is mixed
- What happens when the Query-Answering Agent can't find a grounded answer in the corpus
- What you'd change or add if you had more time/resources

## Part 2: Implementation

Implement the system end to end. It must run on a test set you define, at least 5-10 (article, query/claim) pairs covering all three agents, and produce real outputs.

## Deliverables

- `/Finale/design.md` with your design document
- `/Finale/src/` with your implementation
- `/Finale/test_results/` with outputs on your test set
- `/Finale/README.md` summarizing what you built, what worked, what didn't, and what you'd do differently with more time

## Evaluation

We're looking at:

- Whether your controller routes correctly between the three agents
- How well the Fact-Checker Agent handles ambiguous or conflicting evidence
- Whether your implementation actually runs and produces sensible outputs on your test set
- How well you've documented tradeoffs and limitations

Novel ideation and smart experimentation within this fixed setup (e.g. how you structure routing, how you handle edge cases) are valued over a "safe" implementation that technically works but doesn't show much thought.