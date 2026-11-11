# Digital probabilistic CNNs

Opened 2020-02-10.

A PyTorch harness for probabilistic networks. A run records its setup under `results/`, so an experiment can be reopened later.

Models are in `src/models` (`klmodels.py`, `realmodels.py`). Layers, net parsers, and the training loop sit beside them. `src/bayesgrad.py` is the gradient path for the KL parameterization.

Regularization follows the maximum-probability account: the probability of the trained model, read from the distribution the model defines.

## Papers

- Amir Emad Marvasti, Ehsan Emad Marvasti, Ulas Bagci, and Hassan Foroosh. [Maximum Probability Theorem: A Framework for Probabilistic Machine Learning](https://www.computer.org/csdl/journal/ai/2021/03/09446965/1ua0q5D18Yg). *IEEE Transactions on Artificial Intelligence*, 2021.
- Amir Emad Marvasti, Ehsan Emad Marvasti, G. Atia, Ulas Bagci, and Hassan Foroosh. [Rediscovering Deep Neural Networks Through Finite-State Distributions](https://arxiv.org/abs/1809.10073). *IEEE Access*.
- Amir Emad Marvasti, Ehsan Emad Marvasti, and Ulas Bagci. [Out of Distribution Detection, Generalization, and Robustness Triangle with Maximum Probability Theorem](https://arxiv.org/abs/2203.12145). *ICECCME*, 2022.
