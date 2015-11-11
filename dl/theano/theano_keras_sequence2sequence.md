<!-- toc -->

@(Cabinet)[ml_dl_theano|ml_dl_recurrent|published_gitbook]

# Keras for Sequence to Sequence Learning

> date = "2015-11-10"

Due to my current research projects and Kaggle competition (EEG classification), I'd like to use `keras` for sequence-to-sequence learning. 

> A nice introduction can be found here: http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/


## Regression
According to [*Sequence to sequence learning with neural networks*, NIPS 2014], `sequence2sequence` learning can be seen as an `encoder-decoder` architecture. I'd start from a regression task.

The base code is from http://danielhnyk.cz/predicting-sequences-vectors-keras-using-rnn-lstm/. 

The `X_train` are data points during `i:i+n_prev` time steps, the `y_train` are `2` data points in `i+n_prev` time steps. 

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

n_in_out = 1
n_hidden = 100
n_samples = 2297
n_timesteps = 400

model = Sequential()
# `return_sequences` controls whether to copy the input automatically
model.add(GRU( n_hidden, input_dim = n_in_out, return_sequences=True))
model.add(TimeDistributedDense(n_in_out, input_dim = n_hidden))
model.compile(loss='mse', optimizer='rmsprop')

X = np.random.random((n_samples, n_timesteps, n_in))
Y = np.random.random((n_samples, n_timesteps, n_out))

# learning the hidden states from source sentences

Xp = model._predict(X)
print Xp.shape
print Y.shape

model.fit(X, Y, nb_epoch=10)
```

The most interesting part is the use of `Xp=model._predict(X)`, which acts as the `encoder`. But it may not be necessary. 
