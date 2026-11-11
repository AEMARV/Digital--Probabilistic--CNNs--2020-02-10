# Digital probabilistic CNNs

Opened 2020-02-10.

A PyTorch harness for probabilistic networks. A run records its setup under `results/`, so an experiment can be reopened later.

Models are in `src/models` (`klmodels.py`, `realmodels.py`). Layers, net parsers, and the training loop sit beside them. `src/bayesgrad.py` is the gradient path for the KL parameterization.

Regularization follows the maximum-probability account: the probability of the trained model, read from the distribution the model defines.

The later writeup of that regularization is [Out of Distribution Detection, Generalization, and Robustness Triangle with Maximum Probability Theorem](https://arxiv.org/abs/2203.12145). The finite-state account of a convolutional net is [Rediscovering Deep Neural Networks Through Finite-State Distributions](https://arxiv.org/abs/1809.10073).
