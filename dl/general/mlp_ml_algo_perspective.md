<!-- toc -->

> start: 2015-10-13
> [Machine Learning, An Algorithmic Perspective]
# MLP, Machine Learning An Algorithm Perspective

At this moment, I'm not keen on understanding all the details about how to implement BP algorithm. 


## `np.where`
http://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html

http://stackoverflow.com/questions/13589390/how-to-use-numpy-where-with-logical-operators

http://stackoverflow.com/questions/16094563/numpy-get-index-where-value-is-true

## Data preparation
It is normal to scale the targets to lie between 0 and 1 no matter what kind of activation function is used for the output layer neurons. 

The most common approach to scaling the input data is to treat each data dimension independently, and then to either make each dimension have zero mean and unit variance in each dimension. 

> It is a good idea to normalize the dataset before splitting it into training and testing. 

## Initializing the weights

> [Machine Learning, An Algorithmic Perspective], p80

The MLP algorithm suggest that the weights are initialized to small random numbers, both positive and negative. 

## Shuffling

In a sequential version, the order of the weight updates can matter. It might help to randomize the order of the input data points at `each iteration`. 

```python
change = range(num_data)
np.random.shuffle(change)
inputs = inputs[change, :]
targets = targets[change, :]
```

## Momentum
Imagine a ball rolling down the hill. The reason that the ball stops rolling is because it runs out of energy at the bottom of the dip. If we give the ball some weight, then it will generate momentum as it rolls, and so it is more likely to find the global minimum. We can implement it by adding in some contribution from the previous weight change that we made to the current one. 

## Weight decay
`weight decay` reduces the size of the weights as the number of iterations increases. The argument goes that small weights are better since they lead to a network that is closer to linear (since they are close to zero, they are in the region where the sigmoid is increasing linearly), and only those weights that are essential to the non-linear learning should be large. After each learning iteration through all of the input patterns, every weight is multiplied by some constant within [0, 1]. 

## `np.linspace` problem
Numpy defaults to lists for arrays that are `Nx1`:
```python
>>> x = np.linspace(0, 1, 40)
>>> np.shape(x)
(40,)

>>> x = np.linspace(0,1,40).reshape((1, 40))
>>> np.shape(x)
(1, 40)
>>> np.shape(x.T)
(40,1)
```

