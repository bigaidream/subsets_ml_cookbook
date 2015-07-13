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

A slow option is to automatically vectorizing the function (should really use `NumbaPro` instead):
```python
import numpy as np
H_vec = np.vectorize(H)
```

**Mixing boolean and floating-point calculations**

```python
def H(x):
	return x >= 0
```

**Manual vectorization**

By manual vectorization we normally mean translating the algorithm into a set of calls to functions in the `numpy` package such that no loops are visible in the Python code. It is non-trivial and we should really use `NumbaPro`!

The simple `numpy` recipe (this is similar to the tricks when using GPU to avoid braching) for turning functions of the form:
```python
 def f(x):
	 if condition:
		 r = expression1
	else:
		r = expression2
	return r
```
into vectorized form:
```python
def f_vectorized(x):
	x1 = expression1
	x2 = expression2
	r = np.where(condition, x1, x2)
	return r
```
The `for` loop version is:
```python
def f_for(condition, x1, x2):
	r = np.zeros(len(condition))
	for i in xrange(condition):
		r[i] = x1[i] if condition[i] else x2[i]
	return r
```

Or we can use boolean variables as index:
```python
def Hv(x):
	r = np.zeros(len(x), dtype=np.int)
	r[x >= 0] = 1
	return r
```

### Vectorization of a hat function

The naive Python implementation is
```python
def N(x):
	if x < 0:
		return 0.0
	elif 0 <= x < 1:
		return x
	elif 1 <= x < 2:
		return 2 - x
	elif x >= 2:
		return 0.0
```
The simplest remedy is to use `np.vectorize`:

```python
N_vec = np.vectorize(N)
```

For a manual rewrite, a sketch could be:
```python
if condition1:
	r = expression1
elif condition2:
	r = expression2
elif condition3:
	r = expression3
else:
	r = expression4
```
The replacement treatment is
```python
x1 = expression1
x2 = expression2
x3 = expression3
x4 = expression4
r = np.where(condition1, x1, x4) # first deal with `else`
r = np.where(condition2, x2, r)
r = np.where(condition3, x3, r)
```

Concretely, it is:
```python
def Nv1(x):
	condition1 = x < 0
	condition2 = np.logical_and(0 <= x, x < 1)
	condition3 = np.logical_and(1 <= x, x < 2)
	condition4 = x >= 2

	r = np.where(condition1, 0.0, 0.0)
	r = np.where(condition2, x, r)
	r = np.where(condition3, 2 - x, r)
	r = np.where(condition4, 0.0, r)
	return r
```

Alternatively, we can use boolean indexing, which is also the fastest:
```python
def Nv2(x):
	condition1 = x < 0
	condition2 = np.logical_and(0 <= x, x < 1)
	condition3 = np.logical_and(1 <= x, x < 2)
	condition4 = x >= 2

	r = np.zeros(len(x))
	r[condition1] = 0.0
	r[condition2] = x[condition2]
	r[condition3] = 2- x[condition3]
	r[condition4] = 0.0
	return r
```

### More on numerical Python arrays
`in-place` expression `a += b` is faster than `a = a + b`. 