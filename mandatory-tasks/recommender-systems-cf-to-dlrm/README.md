# Recommender Systems: From Neighbors to DLRM

Welcome to a small journey through the evolution of recommendation systems. Think about opening Netflix after a long day: the platform is not merely asking which movies are similar. It is combining who you are, what you have watched, what is being shown, and the context of this particular moment to estimate what you might click next.

This repository turns that idea into three increasingly capable experiments. We begin with people and items as points in a preference table, first using neighbors and then latent factors. We then teach a neural network to learn nonlinear patterns, and finally assemble those ideas into a Deep Learning Recommendation Model (DLRM)-style system for mixed categorical and numerical data.

## The three-step journey

1. **[Collaborative Filtering](RecommendationTasks/01_CollaborativeFiltering/README.md)**
	 - Compare memory-based recommendations with model-based matrix factorization.
2. **[Neural CTR Model](RecommendationTasks/02_NeuralCTR/README.md)**
	 - Move beyond linear assumptions and learn nonlinear combinations of ad features.
3. **[DLRM](RecommendationTasks/03_DLRM/README.md)**
	 - Combine dense features, categorical embeddings, and explicit feature interactions in a modern recommendation architecture.

Task 01 is open-ended: choose any relevant user-item interaction dataset for the collaborative filtering comparison. Only the final two tasks use the supplied advertising dataset, where each impression contains **13 numeric features**, **26 categorical features**, and a binary `label` showing whether the ad was clicked. The same data lets you compare a simple baseline with a model designed for industrial-scale ranking problems.

## Dataset

See the [dataset guide](datasets/README.md) for more dataset info.

## General instructions

- Explore the data before modelling. Include useful distributions, missing-value checks, class-balance analysis, cardinalities, and interaction statistics.
- Explain every meaningful training and evaluation choice. For example, if you choose Adam over SGD, discuss the expected benefit and the tradeoff.
- Compare methods fairly: use a documented split, avoid test-set tuning, and keep preprocessing learned from training data only.
- Report metrics that fit binary CTR prediction. At minimum include ROC-AUC, PR-AUC, Accuracy, log loss, and a threshold-based metric such as F1 or recall.
- Include a short result table, plots, and a conclusion after each task.
- Keep notebooks, scripts, figures, and reports inside the relevant task folder. Do not dump all outputs into the repository root.
- Use original work. Do not use pretrained recommendation weights for the from-scratch model tasks.
- Make the code reproducible: record seeds, package versions, hardware, and the command or notebook order needed to rerun it.

## Submission checklist

- One notebook or clean script per experiment, with outputs or a clear way to reproduce them.
- A concise report in each task folder explaining approach, results, tradeoffs, and lessons learned.
- A final comparison showing what was gained at every step toward DLRM.
- No credentials, private data, or generated files that cannot be reproduced.
