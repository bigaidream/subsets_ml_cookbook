<!-- toc -->

# Keras for Sequence to Sequence Learning

Due to my current research projects and Kaggle competition (EEG classification), I'd like to use `keras` for sequence-to-sequence learning. 

> The code can be found in https://github.com/ml-playground/keras/tree/master/playground

## Regression
According to [*Sequence to sequence learning with neural networks*, NIPS 2014], `sequence2sequence` learning can be seen as an `encoder-decoder` architecture. I'd start from a regression task.

The base code is from http://www.danielhnyk.cz/blog/view/predicting-sequences-vectors-keras-using-rnn-lstm. However, as in `2015-August-12`, there seems to be some typos and the explanation is a bit confusing. 

> My slightly modified code is here: https://github.com/ml-playground/keras/blob/master/playground/lstm_regression.py

It can be seen from https://github.com/ml-playground/keras/blob/master/playground/lstm_regression.py#L32-L33, that, the `X_train` are data points during `i:i+n_prev` time steps, the `y_train` are `2` data points in `i+n_prev` time steps. 

The data representation techniques can be found here: http://deeplearning.net/tutorial/rnnslu.html#context-window

## Sequence to sequence
 My solution is based on the following two issues of `keras` in github:

> https://github.com/fchollet/keras/issues/314   It's the most useful one.
> https://github.com/fchollet/keras/issues/395

The code is as follows:
```python
from keras.models import Sequential
import numpy as np
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import TimeDistributedDense, Activation

n_in = 1
n_out = 1
n_hidden = 100
n_samples = 2297
n_timesteps = 400

model = Sequential()
model.add(GRU(n_in, n_hidden, return_sequences=True))
model.add(TimeDistributedDense(n_hidden, n_out))
model.compile(loss='mse', optimizer='rmsprop')

X = np.random.random((n_samples, n_timesteps, n_in))
Y = np.random.random((n_samples, n_timesteps, n_out))

# learning the hidden states from source sentences
Xp = model._predict(X)
print Xp.shape
print Y.shape

model.fit(X, Y, nb_epoch=10)
```

The most interesting part is the use of `Xp=model._predict(X)`, which acts as the `encoder`. 