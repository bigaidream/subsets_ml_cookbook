@(Cabinet)[ml_dl_tf, published_gitbook]

> date: 2016-10-24

# Debugging in TensorFlow

<!-- toc -->

Unlike in Torch, the variables in TensorFlow are symbolic by nature. 

Good slides about debugging in TF can be found [here](https://wookayin.github.io/TensorflowKR-2016-talk-debugging/)

The associated code can be found [here](https://github.com/wookayin/TensorflowKR-2016-talk-debugging)

## Basic Approaches

### Single variables
The most simple method is to convert tensors to numpy array and print. 

For example `x` is a TF tensor, 
```python
x_np = sess.run(x)
print(x_np)
```
Here we convert `x` into a numpy object `x_np`. 

Or, if `x` needs some `feed_dict`, we can 
```python
real_input = 199999
x_np = sess.run(x, feed_dict=real_input)
print(x_np)
```

Or we can use `tf.InteractiveSession()` to experiment in shell or Jupyter Notebook. 

## Advanced approaches

TensorBoard is too heavy. 

Use `tf.Assert()` as often as possible. 

