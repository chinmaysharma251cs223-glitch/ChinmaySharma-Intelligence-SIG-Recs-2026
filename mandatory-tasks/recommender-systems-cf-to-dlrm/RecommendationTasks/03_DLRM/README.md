# Task 3: DLRM

This is the destination of the progression. A production recommender may need to combine a user's identity, an ad's identity, campaign context, and dense signals such as counts or time features. DLRM gives each kind of information a natural route: dense features go through an MLP, categorical features become embeddings, and pairwise interactions are learned explicitly before the final prediction.

## Objective

Read and implement the core [DLRM architecture](https://arxiv.org/abs/1906.00091) from scratch. Do not use pretrained recommendation weights or a ready-made DLRM implementation. Reuse the supplied CTR dataset so the comparison with Task 2 is meaningful. The supplied dataset is for Tasks 02 and 03; Task 01 uses an independently selected recommendation dataset.

## Requirements

- Parse the 13 numerical and 26 categorical fields without leaking validation or test information into preprocessing.
- Build embedding tables for categorical fields and an MLP for dense fields.
- Implement the interaction operation described in the paper and combine it with the dense representation for binary click prediction.
- Compare against Task 2 using the same split, seed policy, and metrics.
- Report ROC-AUC, PR-AUC, log loss, and a threshold metric. Include training cost, parameter count, memory considerations, and calibration if possible.
- Run at least one ablation: remove interactions, change embedding dimension, alter the dense MLP, or replace the interaction module with a simpler one.
- Explain what DLRM gains over matrix factorization and the vanilla neural network, and what additional complexity it introduces.

## Deliverables

Include the paper notes, from-scratch implementation, ablation results, final comparison across all three tasks, and a report connecting architecture choices to real recommendation behavior.

## Resources

- [DLRM paper](https://arxiv.org/abs/1906.00091)