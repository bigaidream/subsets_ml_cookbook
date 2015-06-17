# Loop
<!-- toc -->

## Summations
$$\mathrm{sin}(x)\approx x-\frac{x^{3}}{3!}+\frac{x^{5}}{5!}-\frac{x^{7}}{7!}+...$$


> *A primer on Scientific Programming with Python*, p59

```python
x = 1.2
N = 25 # maximum power in the sum
k = 1
s = x
sign = 1.0
import math

while k < N:
    sign = - sign
    k = k + 2
    term = (sign * x**k) / math.factorial(k)
    s = s + term 

print('sin(%g) = %g (approximation with %d terms)'\ 
    % (x, s, N)
```