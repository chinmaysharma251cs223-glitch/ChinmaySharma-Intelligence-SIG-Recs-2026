# Causal Inference

## The Core Idea

Most of machine learning answers "what is correlated with what." Causal inference asks a harder question: if I change X, what happens to Y? The standard framework for this is potential outcomes:

ATE = E[Y(1) − Y(0)]

where:

* Y(1) is the outcome an individual would have if they received the treatment,
* Y(0) is the outcome the same individual would have if they did not,
* ATE is the Average Treatment Effect, the average difference across the population.

The catch: for any individual, you only ever observe one of Y(1) or Y(0), never both, this is called the fundamental problem of causal inference. You can't just compare "people who got the treatment" against "people who didn't" and call the difference the causal effect, because those two groups likely differ in other ways too (confounders), and those differences, not the treatment, might explain the outcome gap.

## Concept of ML

You don't need a background in econometrics for this, just an understanding of what a confounder is (a variable that affects both who gets the treatment and the outcome) and why naive comparison breaks in its presence. In this task you'll explore how ML methods estimate causal effects despite this problem: propensity score methods (modeling who is likely to receive treatment, and reweighting/matching based on that), and more modern causal ML methods like causal forests, which extend tree-based models to estimate how the treatment effect varies across individuals, not just on average.

## Task Instructions

Steps:

* Use the IHDP (Infant Health and Development Program) dataset, a semi-synthetic benchmark widely used in causal inference research. It has real covariates but simulated outcomes, which means you actually have access to ground-truth treatment effects to check your estimates against, something almost no real causal dataset gives you.
* Start with the naive estimate: just compare mean outcomes between the treated and untreated groups directly. Note how far off this is from the true effect (it should be noticeably biased, that's the point).
* Implement propensity score matching or inverse propensity weighting, and re-estimate the treatment effect.
* Implement a causal forest (via the `econml` or `causalml` library) to estimate individual treatment effects (ITEs), not just the average.
* Since IHDP gives you ground truth, plot your estimated ITEs against the true ITEs and report how close you got, this is the actual test of whether your methods worked.
* Check the overlap/positivity assumption, visualize the propensity score distributions for treated vs. untreated groups, and discuss what happens to your estimates in regions with poor overlap.

## Note

IHDP is convenient precisely because it's semi-synthetic, you get ground truth to validate against, which real-world causal problems almost never provide. If you want a harder, more realistic follow-up, try the [Lalonde dataset](https://users.nber.org/~rdehejia/data/.nswdata2.html) (a classic job-training-program dataset), it's real observational data with no ground-truth treatment effect available, so you'll need to reason about whether your estimate is even plausible rather than checking it against an answer key. That gap, between validated methods on synthetic data and blind trust in methods on real data, is the central tension of applied causal inference, and worth sitting with. The [EconML](https://github.com/py-why/EconML) and [CausalML](https://github.com/uber/causalml) libraries are good starting points for implementation either way.
