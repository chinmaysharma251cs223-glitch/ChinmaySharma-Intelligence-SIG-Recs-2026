# Part C: Chaining the Agents into a Pipeline

## Goal

Build a controller that chains the three agents from Part B (Summarizer, Query-Answering, Fact-Checker) into a single working pipeline over the NewsQA corpus. The controller decides which agent(s) to call, in what order, and what to pass between them, given a raw input.

## What You Need to Do

1. Design a controller that takes a raw input (an article, a question, or a claim) and routes it through some combination of the three agents.
2. Decide and justify your routing logic. A simple example flow: given an article and a question, answer the question using the Query-Answering Agent, extract the key claim from that answer, then verify it using the Fact-Checker Agent. You are not required to use this exact flow, design your own if you have a better idea.
3. Handle the case where a step fails or returns low confidence (e.g. what happens if the Fact-Checker Agent finds no supporting evidence, or the Query-Answering Agent can't find a grounded answer). Your controller should degrade gracefully, not just crash or silently ignore the issue.
4. Run your pipeline end to end on at least 5 different inputs (a mix of articles, questions, and claims) and record the full trace (which agents were called, in what order, with what inputs/outputs) for each.

## Encouraged Experimentation

- Conditional routing: not every input needs every agent, can your controller decide which agents are actually needed
- Parallel vs. sequential calls where it makes sense
- Retry logic: if the Fact-Checker Agent returns "unverifiable," can the controller reformulate the claim and try again

Novel routing strategies and smart handling of edge cases are more valuable here than simply chaining all three agents in a fixed order every time.

## Deliverables

- `/PartC/src/` with your controller implementation
- `/PartC/traces/` with the full input-to-output trace for at least 5 example runs, showing which agents were invoked and in what order
- `/PartC/README.md` documenting your routing logic, why you designed it that way, and how you handle failures/edge cases

## Notes

This part is where the "multi-agent" aspect actually comes together, so the writeup matters as much as the code. Explain your reasoning, not just what the code does. This pipeline is effectively a preview of the Finale, so build it in a way you can extend rather than throw away.