<!-- toc -->


# Implementation of GP from Scratch

> Mainly adapted from [*Machine Learning: An Algorithmic Perspective*, 2014]

## Regression

The full Python code is [here](https://github.com/bigaidream/subsets_ml_cookbook/tree/master/bayes/gp/gp_code_scratch/gp.py)

It's desirable to let the optimization process search over different models as well as the parameters of the model. We can generalize the idea of a probability distribution to `stochastic process`, and it is simply a collection of random variables put together: instead of having a set of parameters that specify a probability distribution (such as the mean and covariance matrix for a multivariate Gaussian), we have a set of functions and a distribution over that set of functions. GPs are just smoothers, fitting a smooth curve through a set of datapoints. 

The basic procedure: compute the covariance matrix of the training data, and also the covariances between the training and test data, and the test data alone. Then compute the mean and covariance of the posterior distribution and sample from it. 

### Cholesky decomposition

> [*Machine Learning: An Algorithmic Perspective*, 2014], p402

We use `np.linalg` to decompose a real-valued , symmetric and positive definite matrix $$\mathbf{K}$$ into the product $$\mathbf{LL}^{T}$$, where $$\mathbf{L}$$ is a lower triangular matrix that only has non-zeros entries on and below the leading diagonal. It follows that $$\mathbf{K}^{-1}=\mathbf{L}^{-T}\mathbf{L}^{-1}$$. 

To solve $$\mathbf{LL}^{T}x=t$$, we first use `forward substitution` to find the $$z$$ that solves $$\mathbf{Lz}=\mathbf{t}$$. Then we use `back-substitution` to find the x that solves $$\mathbf{L}^{T}x=z$$. 

> [*Gaussian Processes for Machine Learning*, p37, GP pseudo-code]

The mean, `f`, and covariance, `V` can be computed as
```python
L = np.linalg.cholesky(k)
# `beta` is `alpha` in [GP for ML] book
# `t` is `y` in [GP for ML] book
beta = np.linalg.solve(L.transpose(), np.linalg.solve(L,t))
kstar = kernel(data, xstar, theta, wantderiv=False, measnoise=0)
f = np.dot(kstar.transpose(), beta)
v = np.linalg.solve(L, kstar)
V = kernel(xstar, xstar, theta, wantderiv=False, measnoise=0 - np.dot(v.transpose(), v))
```

### Learning the hyperparameters
$$k(x,x')=\sigma_{f}^{2}\mathrm{exp}(-\frac{1}{2l^{2}}|x-x'|^{2})$$

$$\sigma_{f}$$ is the signal variance, that controls the overall variance of the function; $$\sigma_{n}$$ is a Gaussian noise added into the hidden function value; $$l$$ is the length-scale, that changes the degree of smoothing, trading it off against how well the curve matches the training data.

The squared exponential covariance matrix has three hyperparameters $$(\sigma_{f},\sigma_{n},l)$$ that need to be selected. 

If the set of hyperparameters are labelled as $\theta$ then the ideal solution to this problem would be to set up some kind of prior distribution over the hyperparameters and then integrate them out in order to maximize the probability of the output targets:

$$P(t^{*}|x,t,x^{*})=\int P(t^{*}|x,t,x^{*},\theta)P(\theta|x,t)d\theta$$


> [*Machine Learning: An Algorithmic Perspective*], p404

```python
import scipy.optimize as so
result = so.fmin__cg(logPosterior, theta, fprime=gradLogPosterior, args=[(X,y)],
gtol=1e-4, maxiter=5, disp=1)
```

## Classification
> [*Machine Learning: An Algorithmic Perspective*] sometimes is slack about notations. In case of unclear notations, refer to [Gaussian Processes for Machine Learning*]

To squash the output, `a`, from a regression GP, we use $$P(t^{*}=1|a)=\sigma(a)=1/(1+\mathrm{exp}(-a))$$, where $$\sigma(.)$$ is a logistic function, and $$\sigma_{n}$$ is a hyperparameter and $$\sigma^{2}$$ is the variance. 