# Attention + Decoding Strategies

**Goal:** Extend task 1's encoder-decoder with attention, and compare decoding strategies. Stays within single-sentence translation - no new inputs, no new domain, purely a quality improvement.

**What to do:**
1. Add an attention mechanism of your choice (Bahdanau, Luong, or any variant) on top of Exp 1's decoder, so it can attend to all encoder states instead of relying on one fixed context vector.
2. Re-train and re-evaluate on the same dataset as task 1.
3. Compare decoding strategies at inference
4. Write up **Dataset:** same as task 1 - a direct extension, not a new problem.

**Deliverable:** attention-augmented seq2seq model, BLEU comparison table (no-attention vs attention), and a decoding strategy comparison with example outputs.
