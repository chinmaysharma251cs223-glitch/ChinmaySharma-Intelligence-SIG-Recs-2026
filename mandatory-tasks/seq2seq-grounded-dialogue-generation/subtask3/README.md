# Document-Grounded Dialogue Generation in Hinglish

**Goal:** This is where the task changes shape, not just scales up. task 1-2 solved single-sentence, single-language translation. This experiment moves to multi-turn conversation, two simultaneous conditioning inputs instead of one, code-mixed output, and generation instead of translation. What carries over is the *proven architecture* from task 1-2, not the task itself.

**What's new here:**
- **Two inputs, not one** - the model reads a growing conversation history *and* a separate grounding document at the same time (two encoders, or one encoder run twice with attention split across both sources).
- **Code-mixed, not monolingual** - vocabulary mixes Hindi and English within a sentence, so standard tokenization/embeddings don't apply; this experiment needs its own preprocessing and embedding work.
- **Generation, not translation** - there's no single "correct" reply, so evaluation goes beyond BLEU into topical/contextual judgment.

**What to do:**
1. Data loading + cleaning - parse the CMU Hinglish DoG dataset.
2. Custom preprocessing.
3. Generate embeddings from scratch.
4. Generation model - take the encoder-decoder-with-attention from Exp 2, extend it to encode two context sources (dialogue history + document), decoder attending across both.
5. Baselines - a retrieval-based reply picker, or an ungrounded language-model-only generator, to show the full model actually uses the document meaningfully.
6. Evaluation - BLEU/ROUGE against reference replies, plus qualitative analysis: topical consistency, degradation on heavily code-mixed vs mostly-English turns.

**Dataset:** CMU Hinglish DoG (Document Grounded Conversations) `https://huggingface.co/datasets/festvox/cmu_hinglish_dog`
- Real two-person chat sessions discussing a topic grounded in a Wikipedia document
- Each turn given in both English and Hinglish (`hi_en`) form
- Includes conversation structure (turn order, speaker ID), document indices, session metadata
- Freely downloadable via Hugging Face `datasets`, no login or request needed

**Deliverable:** a working document-grounded Hinglish dialogue generator, a direct comparison against the ungrounded baseline, and a short write-up on any design decisions made along the way.
