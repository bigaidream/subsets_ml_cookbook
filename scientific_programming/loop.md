
<!-- toc -->

# Loop & Lists

## Summation
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

print('sin(%g) = %g (approximation with %d terms)'% (x, s, N)
```

## Changing list elements
The following method does not change the content:
```python
for c in Cdegrees:
	c += 5
```
This changes:
```python
for i in range(len(Cdegrees)):
	Cdegrees[i] += 5
```
Traversing a list. i.e. getting both the index and an element in each pass of the loop can be done with `enumerate`
```python
for i, c in enumerate(Cdegrees):
	Cdegrres[i] = c + 5
```

## Traversing nested lists
> [*A primer of scientific programming with Python*], p74

Suppose we use a nested list `scores` to record the scores of players in a game: `scores[i]` holds a list of the historical scores of player number `i`. Different players have played the game a different number of times, so the length of `scores[i]` depends on `i`.
```python
scores = []
# score of player no. 0
scores.append([12, 16, 11, 12])
# score of player no. 1
scores.append([9])
# score of player no. 2
scores.append([6, 9, 11, 14, 17, 15, 14, 20])
```

The index-based version of traversing is
```python
for p in range(len(scores)):
	for g in range(len(scores[p])):
		score = scores[p][g]
		print('%4d' % score,)
	# adds a newline after each table row
	print
```

A Pythonic version is

```python
for player in scores:
	for game in player:
		print('%4d' % game,)
	print
```

