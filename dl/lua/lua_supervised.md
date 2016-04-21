<!-- toc -->

# Supervised Learning with Torch/Lua

Date = "2016-01-30"

## Why Torch/Lua?

I still love Python, but it's really painful to debug in Theano. I will probably rewrite my Python code used in http://arxiv.org/abs/1601.00917 using Torch/Lua. Also there are some good reinforcement learning libraries written in Torch/Lua. 

~~I prefer Eclipse for Lua/Torch. It seems that for Eclipse to debug Luajit data correctly, we have to use `require 'debugger.plugins.ffi'`.~~

> '2016-03-12', it turns out that Intellij IDEA is better. For Lua code with Torch, Eclipse sometimes cannot set breakpoints successfully. My current best practice is to write code in Intellij IDEA and use [fblualib-debugger](https://github.com/facebook/fblualib/tree/master/fblualib) to debug. 

This note is based on: https://github.com/torch/tutorials/tree/master/2_supervised

## Data

 > http://code.madbits.com/wiki/doku.php?id=tutorial_supervised_1_data

Global variables are used a lot in the tutorials. 

`slicing` operation tutorial can be found here: https://github.com/torch/tutorials/blob/master/2_supervised/A_slicing.lua

Compared to Python, Lua cannot slice arbitrary subtensors
```lua
require 'torch'
t2 = torch.range(1, 25):resize(5,5)
t4 = torch.Tensor(5, 2)
t4[{ {}, 1}] = t2[{ {}, 2}]
t4[{ {}, 2}] = t2[{ {}, 5}]
```

## Model
 > http://code.madbits.com/wiki/doku.php?id=tutorial_supervised_2_model

The construction of models is pretty similar to Keras. In fact, Torch is the teacher of Keras. 

One neat way to get the weights is:
```lua
model:get(1)
```
where the meaning of numbers can be seen by `print(model)`. 

## Loss Function
Mean-square error (MSE) loss is typically not a good choice for classification, as it forces the model to exactly predict the values imposed by the targets. 

Instead, we prefer a probabilistic objective -- the negative log-likelihood. 

```lua
model:add( nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
```

## Training

>http://code.madbits.com/wiki/doku.php?id=tutorial_supervised_4_train

In practice, it's very important to start with a few epochs of pure SGD, before switching to L-BFGS or ASGD (if switching at all). The intuition for that is related to the non-convex nature of the problem: at the very beginning of training (random initialization), the landscape might be highly non-convex, and no assumption should be made about the shape of the energy function. Often, SGD is the best we can do. Later on, batch methods (L-BFGS, CG) can be used more safely. 

The `torch.optim` module can be found here: https://github.com/torch/optim

## Testing

> http://code.madbits.com/wiki/doku.php?id=tutorial_supervised_5_test

## Tips

### Nonlinearity
Nonlinearities that are symmetric around the origin are preferred because they tend to produce zero-mean inputs to the next layer (which is a desirable property). Empirically, the `tanh` has better convergence properties. 

### Weight initialization
At initialization we want the weights to be small enough around the origin so that the activation function operates near its linear regime, where gradients are the largest. Otherwise, the gradient signal used for learning is attenuated by each layer as it is propagated from the classifier towards the inputs. Each module has a `reset()` method, which initializes the parameter with a uniform distribution that takes into account the fanin/fanout of the module. 

There is a separate module for this: https://github.com/Kaixhin/nninit

### Number of hidden units
The number of hidden units that gives best results is dataset-dependent. Generally speaking, the more complicated the input distribution is, the more capacity the network will require to model it, and so the larger the number of hidden units that will be needed.

### Norm Regularization
Typical values to try for the L1/L2 regularization parameter are 10^-2 or 10^-3. It is usually only useful to regularize the topmost layers of the MLP (closest to the classifier), if not the classifier only.

### ConvNet Training
#### Number of filters
Since feature map size decreases with depth, layers near the input layer will tend to have fewer filters while layers higher up can have much more. In fact, to equalize computation at each layer, the product of the number of features and the number of pixel positions is typically picked to be roughly constant across layers. To preserve the information about the input would require keeping the total number of activations (number of feature maps times number of pixel positions) to be non-decreasing from one layer to the next (of course we could hope to get away with less when we are doing supervised learning). The number of feature maps directly controls capacity and so that depends on the number of available examples and the complexity of the task.

#### Filter Shape
Common filter shapes found in the literature vary greatly, usually based on the dataset. Best results on MNIST-sized images (28x28) are usually in the 5x5 range on the first layer, while natural image datasets (often with hundreds of pixels in each dimension) tend to use larger first-layer filters of shape 7x7 to 12x12.

The trick is thus to find the right level of "granularity" (i.e. filter shapes) in order to create abstractions at the proper scale, given a particular dataset.

#### Pooling Shape
Typical values for pooling are 2x2. Very large input images may warrant 4x4 pooling in the lower-layers. Keep in mind however, that this will reduce the dimension of the signal by a factor of 16, and may result in throwing away too much information. In general, the pooling region is independent from the stride at which you discard information. In Torch, all the pooling modules (L2, average, max) have separate parameters for the pooling size and the strides, for example:

```lua
nn.SpatialMaxPooling(pool_x, pool_y, stride_x, stride_y)
```
