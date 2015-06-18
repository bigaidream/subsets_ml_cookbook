<!-- toc -->

# GPy tutorial

## Basics
[Very basics of GPy, interacting with models, ipynb](http://nbviewer.ipython.org/github/SheffieldML/notebook/blob/master/GPy/models_basic.ipynb)

## Regression tutorial
[GP regression by Nicolas Durrande 2013, ipynb](http://nbviewer.ipython.org/github/SheffieldML/notebook/blob/master/GPy/basic_gp.ipynb)

```python
import numpy as np
from matplotlib import pyplot as plt
import GPy

X = np.random.uniform(-3.,3.,(20,1))
# add noise into Y
Y = np.sin(X) + np.random.randn(20,1)*0.05
```
One trick when working with `PyCharm` is to disable `interactive` functionality in `matplotlib` as 
```python
plt.interactive(False)
```

Define the covariance kernel, i.e. RBF, and then form the GPy model `m`. 
```python
kernel = GPy.kern.RBF(input_dim=1, variance = 1., lengthscale= 1.)
m = GPy.models.GPRegression(X, Y, kernel)
```
After initialization, we can optimize
```python
# the normal way
# m.optimize(messages=True)
# with restarts to get better results
m.optimize_restarts(num_restarts = 20)
```

## Kernel
[Kernel tutorial by Nicolas Durrande, 2013, ipynb](http://nbviewer.ipython.org/github/SheffieldML/notebook/blob/master/GPy/basic_kernels.ipynb)

### Combine kernels
```python
k1 = GPy.kern.RBF(1, 1., 2.)
k2 = GPy.kern.Matern32(1, 0.5, 0.2)

# product of kernels
k_prod = k1 * k2
k_prod.plot()

# Sum of kernels
k_add = k1 + k2
k_add.plot()
```
The kernels that have been added are pythonic in that the objects remain linked: changing parameters of an add kernel changes those of the constituent parts, and vice versa
```python
k_prod.rbf.variance = 12.
print k1
```


