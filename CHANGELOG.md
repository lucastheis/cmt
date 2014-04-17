# CMT

## 0.5.0

- Regularizers are now more flexible.
- Several bug fixes and stability improvements.
- Added binomial distribution.
- Added StackedAffineTransform.

## 0.4.1

- Added multinomial logistic regression (MLR).
- Added function for sampling images conditioned on component labels.
- Extended MCGSM by additional parameters.

## 0.4.0

- Added spike-triggered mixture model (STM).
- Added simple univariate distributions such as Bernoulli and Poisson.
- Added generalized linear model (GLM) and fully-visible belief network (FVBN).
- Added mixture of Gaussian scale mixture (MoGSM).
- Added *PatchMCGSM*.
- Extended MCGSM by linear features and means.
- Made implementation of new conditional models easier by introducing interface *Trainable*.
- Most methods of *MCGSM* can now cope with zero-dimensional inputs.

## 0.3.0

- Implemented *early stopping* based on validation error.

## 0.2.0

- Implemented mixture of conditional Boltzmann machines (MCBM).
- Implemented *PatchMCBM*.
- Extended tools for generating data from images and videos.
