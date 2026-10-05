# Basic Sequence-to-Sequence Translation

**Goal:** Build the foundational encoder-decoder architecture that later experiments extend.

**What to do:**
1. Implement a basic RNN/LSTM-based encoder-decoder from scratch - the encoder compresses the source sentence into a context vector, the decoder generates the target sentence from it, one token at a time.
2. Train it on a parallel English→target-language corpus.
3. Evaluate using BLEU score on a held-out test split.
4. Write up: how well the basic model performs, and where it visibly struggles (e.g. long sentences, rare words) - this motivates Exp 2's improvements.

**Dataset:**
- https://drive.google.com/drive/folders/1R4pOtvDEYyXtjLgVsT2RsmaORKOzoEq2?usp=drive_link


**Deliverable:** a working basic seq2seq model, BLEU score, and a short note on its failure modes.
