# Efficient Causal Convolutions for Time-Series Forecasting

[![codecov](https://codecov.io/gh/stefanocampanella/devtools_scicomp_project_2025/graph/badge.svg?token=VOWYD2SB14)](https://codecov.io/gh/stefanocampanella/devtools_scicomp_project_2025)

In this project, an autoregressive model (ARM) using causal convolutions has been implemented. The purpose is to 
showcase the usage of software engineering practices (e.g. unit tests), development tools (e.g. git), and array 
computation libraries commonly used in deep-learning (e.g. JAX).

The code is a complete rewrite in JAX of one of the assignments given at the [2024th edition of the Generative Modelling 
Summer School](https://gemss.ai/2024/) at the Eindhoven TUe. 

## Introduction

An ARM is a likelihood-based deep generative model that is parameterized by causal convolutional neural networks or 
causal transformers. Here, we focus on causal convolutions. The approach of parameterizing ARMs with causal 
convolutions was utilized in multiple papers, e.g.:
- [Van den Oord, Aaron, et al. "Conditional image generation with pixelcnn decoders." NeurIPS 29 (2016).](https://proceedings.neurips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders)
- [Oord, Aaron van den, et al. "Wavenet: A generative model for raw audio." arXiv preprint arXiv:1609.03499 (2016).](https://arxiv.org/abs/1609.03499)

You can read more about ARMs in Chapter 2 of the following book:
- [Tomczak, J.M., "Deep Generative Modeling", Springer, 2022](https://link.springer.com/book/10.1007/978-3-030-93158-2) 

ARMs are probabilistic models that utilize factorization of joint distribution in the following manner:

```math
p(x) = \prod_{d=1}^{D - 1} p(x_d | x_{1:d-1}) \; p(x_0).
```

Then the log-likelihood function is then the following:

```math
\ln p(x) = \sum_{d=1}^{D - 1} \ln p(x_d | x_{1:d-1}) + p(x_0).
```

This is very convenient because we *only* need to model conditionals. The challenge though is how to allow learning 
long-range dependencies (a *long-term memory*). One possible way of accomplishing that is by utilizing 
**causal convolutions**. Then, we can use a convolutional neural network that predicts the parameters of the 
conditionals in a single forward run.

The experiments performed in this repo will be performed on the **sequential MNIST** dataset.

## Causal Convolutions

One dimensional causal convolutions are just regular convolutions on an array of values $(x_0, \dots, x_{D-1})$ where 
the $k$-th element of the result dependent only on the previous 
$(x_0, \dots, x_{k - 1})$ or $(x_0, \dots, x_{k - 1})$ values.

This means that kernel convolutions with weights $(w_0, \dots, w_{n - 1})$ are not centered around the current value as 
usually do, i.e. the causal convolution operator $C$ is such that

```math
C(x, w)_k = \sum^{n - 1}_{i = 0} w_{i} x_{k - n + i + (1 - A)} \; ,
```

where $A = 1$ for the case of dependency on current token or $A = 0$ if not. 

In the implementation this can be obtained by using carefully chosen padding and slicing of input and output arrays. 
Furthermore, in case of dilation one must take that into account when padding.

## Dataset and Probability Distribution Parametrization

The MNIST dataset contains $28\times28$ images of handwritten digits, where the value of each pixel is quantized to 
fixed brightness levels in the range $\{0, \cdots, 255 \}$. 

Hence, a natural choice is to use a categorical 
distribution for each pixel $x_d$, i.e. a probability mass function $\theta_{dx}$ for $j$ taking values in $\{0, \cdots, 255\}$.

Then the log-probability of a sequence $(x_0, \dots, x_{D - 1})$, after taking the factorization of the 
joint distribution using the product rule into account and up to a constant, will be

```math
\log{p} = 
\sum^{D - 1}_{d = 0} \log{\left(\theta_{d x_d}\right)} = 
\sum^{D - 1}_{d = 0} \sum^{255}_{i = 0} \delta_{i x_d} \log{(\theta_{d i})}
```

where $\delta_{i j}$ is the Kronecker delta.

After the repeated application of the product rule, the probabilities $\theta_{d i}$ are functions of the sampled 
values up to $x_{d - 1}$, i.e. $\theta_{d i} = \theta_{d i}\left(x_0, \cdots, x_{d - 1}\right)$. One can start from a 
fixed value for $x_0$, say $x_0 = 0$, and then sample $x_1$ according to prescribed categorical distribution, which 
can be sampled using `jax.random.categorical`, and so on.