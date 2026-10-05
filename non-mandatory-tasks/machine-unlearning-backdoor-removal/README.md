# Machine Unlearning: Making a Model Forget a Backdoor

Imagine you trained a model, and later found out that 500 of your 50,000 training images were tampered with. Someone had stamped a small mark on them and given them the wrong label, so the model would misbehave whenever it saw that mark. Retraining from scratch would fix this, but it's too expensive, too slow, or impractical.

This task asks a simple question: **can you remove the effect of just those 500 bad images, without retraining**

## Learn the Basics First

If machine unlearning is new to you, start here before writing any code:
- 🎥 Video: [How does machine unlearning work?](https://www.youtube.com/watch?v=C-k4Zf39nNg) - a short, simple explanation.
- 📖 Article: [Machine Unlearning (Wikipedia)](https://en.wikipedia.org/wiki/Machine_unlearning) - a clear overview of why it matters and how it's done.
- 📄 Paper (deeper dive, optional): [Machine Unlearning: Solutions and Challenges](https://arxiv.org/abs/2308.07061) - a survey covering most known methods.

## Dataset

**CIFAR-10** - https://www.cs.toronto.edu/~kriz/cifar.html (also available as `torchvision.datasets.CIFAR10`). 50,000 training images, 10 classes, 32×32 pixels, in color.

## What You Need to Build

1. **Pick a target label:** class 9, `truck`.
2. **Make a simple trigger:** a 4×4 pure white square, placed in the bottom-right corner of the image (pixels `[28:32, 28:32]`, set to RGB `(255, 255, 255)`). In code, this just means overwriting those pixel values directly on the image array.
3. **Pick 500 images to poison:** choose 500 images at random from the 9 classes that are *not* `truck` (1% of the training set).
4. **Poison them:** stamp the white square on each of these 500 images, and change their label to `truck`. Leave the other 49,500 images exactly as they are.

You now have a training set of 50,000 images: 49,500 clean, and 500 poisoned.
- The **500 poisoned images = your forget set**.
- The **other 49,500 images = your retain set**.

## Stage 1 - Train the Model You'll Need to Fix

- Train a CNN of your choice on the poisoned training set (all 50,000 images).
- **Check clean accuracy:** test the model on the normal (untouched) test set.
- **Check the Attack Success Rate (ASR):**  this is the key number for this whole task, so here's exactly how to compute it:
  1. Take a batch of **clean, unmodified** test images from the 9 non-truck classes (their real labels don't matter for this check).
  2. Stamp the **same white square** onto each of these images, in the same spot as before.
  3. Feed these newly-triggered images into your model and record its predictions.
  4. **ASR = the percentage of these predictions that come out as `truck`.**
- Also train a second model: a **retrain-from-scratch model** using only the 49,500 clean images (no poison at all). This is your target: what a model looks like when it never saw the bad data.

**Deliverables:** the poisoned model (with clean accuracy + ASR), the retrain-from-scratch model (with clean accuracy + ASR), both saved as checkpoints.

## Stage 2 - Remove the Backdoor Without Retraining

Try some of the methods below:

- **Fine-tune on the retain set.** Keep training your poisoned model, but only ever show it the 49,500 clean images, never the 500 poisoned ones again. Use a small learning rate and just a few epochs. The idea: by only reinforcing correct patterns, the backdoor association may fade, though it might not fully disappear. 📖 [Hands-on tutorial with code](https://pub.towardsai.net/machine-unlearning-a-hands-on-tutorial-with-pytorch-dd75eb9bed7b)
- **Gradient ascent on the forget set.** Take your poisoned model, and run a few training steps on just the 500 poisoned images but flip the direction: instead of *decreasing* the loss like normal training, deliberately *increase* it. This pushes the model away from its habit of predicting `truck` whenever it sees the trigger. 📖 [A Practical Guide to Machine Unlearning: Forgetting by Gradient Ascent](https://medium.com/@zachariaharungeorge/a-practical-guide-to-machine-unlearning-forgetting-by-gradient-ascent-54a07faccc0c)

- **Fisher forgetting** : carefully scaled noise is added to the model's weights, using the Fisher Information Matrix to figure out which weights matter most for the forget set (and leaving the rest alone). 📄 [Golatkar et al., 2020](https://arxiv.org/abs/1911.04933)
- **SCRUB** - train a copy of your model (the "student") to keep agreeing with the original model (the "teacher") on the retain set, while deliberately disagreeing with it on the forget set. 📄 [Kurmanji et al., 2023](https://arxiv.org/abs/2302.09880)

You need **at least two methods total**, the first two are enough to complete this stage and the last two are bonus methods.

**Deliverables:** working code for each method you try, and 2-3 simple sentences per method on why it should remove the backdoor.

## Stage 3 - Prove the Backdoor Is Actually Gone

- For each unlearned model, report **clean accuracy** and **ASR** (using the exact same procedure as Stage 1). Compare these to your poisoned model and your retrain from-scratch model.
- Run a simple **membership inference attack** on the 500 forget-set images. The idea: a model is usually a little more "confident" (lower loss, higher softmax score) on images it was trained on than on images it's never seen. An attacker exploits this gap to guess whether a given image was in the training set. Run this attack against your forget-set images before and after unlearning. 📄 [Yeom et al., 2018](https://arxiv.org/abs/1709.01604) (for a stronger version of this attack: 📄 [LiRA, Carlini et al., 2022](https://arxiv.org/abs/2112.03570))
- **Relearning test:** take your best unlearned model, and fine-tune it on just 10 of the 500 poisoned images to check if the backdoor is completely erased or just hidden.

**Deliverables:** one table comparing (poisoned model, each unlearned model, retrain-from-scratch model) on clean accuracy, ASR, and membership inference success. Plus your relearning test result, explained in a few sentences.

## Stage 4 - Cost and Write-Up

- Time each unlearning method, and compare it to how long the full retrain from scratch took.
- Write a short report answering: which method actually removed the backdoor, and which one only hid it? Use your Stage 3 relearning test as your main evidence.

**Deliverables:** a time comparison and a short written report.

## Submission

- A notebook or GitHub repo with your code and your results.
- A short report for each Stage.

**Good luck!** 🚀
