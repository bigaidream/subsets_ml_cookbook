<!-- toc -->

# Implementation of GP from Scratch

> Mainly adapted from [*Machine Learning: An Algorithmic Perspective*, 2014]

## Regression

It's desirable to let the optimization process search over different models as well as the parameters of the model. We can generalize the idea of a probability distribution to `stochastic process`, and it is simply a collection of random variables put together: instead of having a set of parameters that specify a probability distribution (such as the mean and covariance matrix for a multivariate Gaussian), we have a set of functions and a distribution over that set of functions. GPs are just smoothers, fitting a smooth curve through a set of datapoints. 

The basic procedure: compute the covariance matrix of the training data, and also the covariances between the training and test data, and the test data alone. Then compute the mean and covariance of the posterior distribution and sample from it. 

### Cholesky decomposition

> [*Machine Learning: An Algorithmic Perspective*, 2014], p402

We use `np.linalg` to decompose a real-valued , symmetric and positive definite matrix $\mathbf{K}$ into the product $\mathbf{LL}^{T}$, where $\mathbf{L}$ is a lower triangular matrix that only has non-zeros entries on and below the leading diagonal. It follows that $\mathbf{K}^{-1}=\mathbf{L}^{-T}\mathbf{L}^{-1}$. 

To solve $\mathbf{LL}^{T}x=t$, we first use `forward substitution` to find the $z$ that solves $\mathbf{Lz}=\mathbf{t}$. Then we use `back-substitution` to find the x that solves $\mathbf{L}^{T}x=z$. 

> [*Gaussian Processes for Machine Learning*, 37, GP pseudo-code]

The mean, `f`, and covariance, `V` can be computed as
```python
L = np.linalg.cholesky(k)
# `beta` is `alpha` in GP for ML book
beta = np.linalg.solve(L.transpose(), np.linalg.solve(L,t))
kstar = kernel(data, xstar, theta, wantderiv=False, measnoise=0)
f = np.dot(kstar.transpose(), beta)
v = np.linalg.solve(L, kstar)
V = kernel(xstar, xstar, theta, wantderiv=False, measnoise=0 - np.dot(v.transpose(), v))
```

