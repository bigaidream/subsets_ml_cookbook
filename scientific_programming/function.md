<!-- toc -->

# Function and Branching

## Changing global variables inside functions

The values of global variables can be accessed inside functions, but the values cannot be changed unless the variable is declared as `global`:

```python
a = 20
b = -2.5

def f1(x):
    a = 21
    return a*x + b
    
print(a)

def f2(x):
    global a
    a = 21
    return a*x + b

f1(5)
print(a)

f2(5)
print(a)
```

## Computing sums

To calculate the sum:

$$L(x;n)=\sum_{i=1}^{n}\frac{1}{i}\left(\frac{x}{1+x}\right)^{i}$$

We can use

```python
def L(x, n):
	s = 0
	# the total number is n, but we use n+1 in `range`
	for i in range(1, n+1):
		s += (1.0/i)(x/(1.0 + x))**i
	return s
```


## Functions as arguments to functions

Consider a function for computing the second-order derivative of a function numerically:

$$f"(x)\approx\frac{f(x-h)-2f(x)+f(x+h)}{h^{2}}$$

where $$h$$ is a small number. The Python code is:
```python
def diff2nd(f, x, h=1E-6):
	r = (f(x-h) - 2*f(x) + f(x+h))/float(h*h)
	return r

def g(t):
	return t**(-6)

t = 1.2
d2g = diff2nd(g, t)
```
