<!-- toc -->

# Array Computing and Curve Plotting

## Array
When we need `n` elements with uniformly distributed values in an interval [p,q], The `numpy` function `linspace` works like:
```python
a = np.linspace(p, q, n)
```

-----
List comprehensions result in scalar code because they still have explicit, slow Python `for` loops operating on scalar quantities. But we could use `NumbaPro` to make it much faster. 

Most Python functions intended for a scalar argument `x`, like
```python
def f(x):
	return x**4*exp(-x)
```
automatically work for an array argument `x`:
```python
x = np.linspace(-3, 3, 101)
y = f(x)
```

## Curve plotting
The `Matplotlib` developers do NOT promote the `matplotlib.pylab` interface. Instead, they recommend the `matplotlib.pyplot` module like:
```python
import numpy as np
import matplotlib.pyplot as plt
```
## Advanced vectorizaiton of functions
The heaviside function in Python is:
```python
def H(x):
	return (0 if x < 0 else 1)
```
However, trying to call `H(x)` with an `array` argument `x` fails. It's due to the test `x < 0`, which results in an array of boolean values, while the `if` test needs a single boolean value. There are 4 ways:

**Loop**
Using an explicit `for` loop:
```python
def H_loop(x):
	r = np.zeros(len(x))
	for i in xrange(len(x)):
		r[i] = H(x[i])
	return r
# Example:
x = np.linspace(-5, 5, 6)
y = H_loop(x)
```

**Automatic vectorization**


