# From Translation to Grounded Dialogue Generation

## Overview

Your task is to build sequence-to-sequence NLP systems from the ground up, starting from a basic encoder-decoder for translation and ending with a document-grounded dialogue generator in Hinglish (Hindi-English code-mixed text).

The goal is to first prove out the core encoder-decoder + attention mechanism on a simple, well-understood problem, and then push the same architecture into a genuinely harder, real-world setting involving:

- Multi-turn conversation
- Two simultaneous conditioning inputs
- Document grounding
- Code-mixed Hinglish generation
- Different decoding strategies

The task should demonstrate not only that your model can generate text, but that you understand what is happening inside the encoder, decoder, attention mechanism, and conditioning pipeline.

## Important Guidelines

- **No fine-tuning of large pretrained transformers anywhere in this task**, and **no packaged seq2seq/translation pipelines**.
- You may use standard framework building blocks: `nn.LSTM`/`nn.GRU` cells, `nn.Embedding` layers, optimizers, standard training loops - the constraint is on not importing an already-assembled seq2seq/translation/attention solution, not on avoiding your deep learning framework entirely.
- You are free to use any attention variant and any decoding strategy - experiment freely.
- If you are feeling super enthusiastic, you are free to implement additional architectures (e.g. a from-scratch Transformer) for comparison.

**NOTE:** You are not limited to the datasets suggested in each folder as long as it fits the guidelines.

## Structure

See each folder's own README for the specific goal, what to do, and dataset.

## Resources

- CampusX
- Krish Naik
- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Bahdanau vs Luong Attention Explained](https://towardsdatascience.com/understanding-attention-mechanism-5d6e22a89f7f)
- [https://medium.com/@yashwanths_29644/deep-learning-series-15-understanding-bahdanau-attention-and-luong-attention-773216197d1f](https://medium.com/@yashwanths_29644/deep-learning-series-15-understanding-bahdanau-attention-and-luong-attention-773216197d1f)
- [Helsinki-NLP MarianMT Models](https://huggingface.co/Helsinki-NLP)
- [Neural Machine Translation - Towards Data Science](https://towardsdatascience.com/neural-machine-translation-15ecf6b0b)



## Submission

Submit:

- A clean Colab / Jupyter notebook or GitHub repo with your code and results.
- A short report (2-3 pages) summarizing:
  - Approach & dataset(s)
  - Experimental results
  - Insights and takeaways

Good luck!