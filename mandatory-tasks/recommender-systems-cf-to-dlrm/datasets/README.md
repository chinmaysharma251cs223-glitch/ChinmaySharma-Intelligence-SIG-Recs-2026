# Dataset Guide

The dataset in this folder is provided only for Tasks 02 and 03. Task 01 is open-ended and requires choosing a relevant user-item interaction dataset for the collaborative filtering comparison; it does not require this advertising dataset.

The supplied `dataset (tasks 2 and 3).zip` is an advertising interaction dataset for binary click-through-rate prediction. Each row represents an ad impression and the `label` records whether the user clicked it.

## Contents

- 13 numerical integer features
- 26 categorical features
- 1 binary target: `label` (`0` means no click, `1` means click)
- Separate training and testing files

## Remarks

- Keep the original test file untouched.
- Note that the same dataset is intentionally shared by Tasks 02 and 03. This makes the architecture comparison fair: improvements should come from the model and its feature handling, not from changing the data split.