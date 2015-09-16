<!-- toc -->

> Start: 2015-09-16

# Lasagne Tutorial

As of *16 September 2015*, Nolearn only supports Lasagne, though it's trying to support Keras due to GitHub issue discussion. I need the `cross_validation` and `random_hyper` components of Sklearn to work with deep neural networks. 

Keras hides Theano from users; whereas Lasagne exposes Theano to users to some extent. As far as I know, when training RNNs, I don't have to use `theano.scan()` manually in Lasagne, which is good enough. 

## MLP Example

> http://lasagne.readthedocs.org/en/latest/user/tutorial.html
> The sample code is in the `/examples` folder

`input_var` is the Theano variable that the network's input layer will be linked to. In the example, `input_var` is linked to a variable given as an argument to the `build_mlp()` function. 

After loading the input data, it applies 20% dropout (this is the first time I know this trick):
```python
l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
```

Lasagne does not support `.yaml`, because it believes that is's more flexible to encode architecture in Python itself. 

## CNN
`dropout` tends not to work well for convolutional layers. 