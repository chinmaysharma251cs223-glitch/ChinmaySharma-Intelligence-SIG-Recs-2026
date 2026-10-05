# Neural ODEs

## The Governing Equation

Consider a simple dynamical system, take the damped pendulum:

d²θ/dt² + b (dθ/dt) + (g/L) sin(θ) = 0

or, if you'd rather work with a coupled two-species system, the Lotka-Volterra predator-prey equations:

dx/dt = αx − βxy dy/dt = δxy − γy

where:

* θ(t) is the pendulum's angular displacement, b is the damping coefficient, g is gravity, L is the pendulum length,
* x(t) and y(t) are prey and predator populations, α, β, δ, γ are interaction rates.

Pick either system (or propose your own low-dimensional ODE, as long as it has genuine nonlinear dynamics, not something trivially linear).

## Concept of ML

A standard neural network is a stack of discrete layers, layer 1 to layer 2 to layer 3, and so on. A Neural ODE reframes this: instead of discrete layers, treat the "depth" of the network as continuous time, and learn a function f(z, t, θ) that describes how a hidden state z evolves continuously, dz/dt = f(z, t, θ). You then use a standard ODE solver (Euler, RK4, adaptive solvers) to integrate this learned function forward and get your output.

You don't need to derive or hand-solve the ODE yourself, that's the point, the network learns the dynamics from data. What you do need is a basic grip on what a derivative and an integral represent, and enough comfort with the chosen system's equations to generate data from it and sanity-check what the model produces.

## Task Instructions

Steps:

* Simulate your chosen system (pendulum or predator-prey) numerically to create a synthetic dataset of trajectories, vary the system's parameters (damping coefficient, or the interaction rates) across different trajectories.
* Visualize a handful of trajectories and describe how they change as you vary the parameters.
* Create a grouped train/test split where the parameter values used in training do not appear in testing at all, this is what actually tests whether your model learned the underlying dynamics rather than memorizing specific trajectories.
* Train a Neural ODE model on the training trajectories, and train a standard discrete-time baseline (e.g. an RNN or LSTM) on the same data for comparison.
* Compare both models' ability to extrapolate: predict the trajectory well beyond the time range they were trained on. This is where Neural ODEs are expected to have a real edge.
* Add noise to your training data and see how both models degrade, and what changes (architecture, solver tolerance, regularization) help the Neural ODE cope better.

## Note

This task, like most synthetic-data tasks, is a controlled sandbox, real dynamical systems rarely come with such clean, densely-sampled data. If you want to push further, real-world time series are often irregularly sampled (measurements taken at uneven intervals), which is exactly the kind of data Neural ODEs handle naturally and RNNs handle awkwardly. A good real dataset to try this on is the [PhysioNet Challenge 2012 dataset](https://physionet.org/content/challenge-2012/1.0.0/), ICU patient time series with irregular measurement intervals. The [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) library (from the original Neural ODE paper's authors) is the standard starting point for implementation.

Neural ODEs show up anywhere you have continuous underlying dynamics and want a model that respects that, physiological signals, climate data, financial time series with irregular trades, and physics simulations more broadly. Hope this gives you a new lens on what "depth" in a neural network can even mean.