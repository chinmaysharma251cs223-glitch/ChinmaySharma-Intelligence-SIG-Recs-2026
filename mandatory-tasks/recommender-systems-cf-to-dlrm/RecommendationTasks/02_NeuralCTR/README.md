# Task 2: Neural CTR Prediction

Real recommendation systems rarely see only a user and an item. They also see context: device, time, placement, campaign, and a mixture of numerical and categorical descriptors. This is the step where our toy preference table starts to resemble an ad-ranking system deciding what to show in a split second.

## Dataset

Use `dataset (tasks 2 and 3).zip`, found in the `datasets/` folder at the root of this task. It contains separate training and testing files for binary click-through-rate prediction: 13 integer numerical features, 26 categorical features, and the `label` target.

This supplied advertising dataset is intended for Tasks 02 and 03. It is not required for Task 01, which uses an independently selected recommendation dataset.

## Objective

Train a vanilla neural network baseline that consumes encoded categorical and numeric features. Try several small architectures and select one using the validation set, not the test set.

## Requirements

- Try different neural architectures.
- Report ROC-AUC, Accuracy, PR-AUC, log loss, and a justified threshold metric such as F1, precision, or recall. Include calibration or a reliability plot if possible.
- Track training/validation curves and investigate overfitting.
- Bonus: implement the [Deep & Cross Network](https://arxiv.org/abs/1708.05123) and compare explicit crosses against the vanilla network.

## Deliverables

Submit code, preprocessing decisions, an architecture comparison and metric plots.

## Resources

- [Deep & Cross Network](https://arxiv.org/abs/1708.05123)
