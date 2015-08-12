# Keras for Regression and Time-Series

Due to my current research projects and Kaggle competition (EEG classification), I'd like to use `keras` for regression and time-series data analysis. 

> The code can be found in https://github.com/ml-playground/keras/tree/master/playground

## Regression
The base code is from http://www.danielhnyk.cz/blog/view/predicting-sequences-vectors-keras-using-rnn-lstm. However, as in `2015-August-12`, there seems to be some typos and the explanation is a bit confusing. 

> My slightly modified code is here: https://github.com/ml-playground/keras/blob/master/playground/lstm_regression.py

It can be seen from https://github.com/ml-playground/keras/blob/master/playground/lstm_regression.py#L32-L33, that, the `X_train` are data points during `i:i+n_prev` time steps, the `y_train` are `2` data points in `i+n_prev` time steps. 

The data representation techniques can be found here: http://deeplearning.net/tutorial/rnnslu.html#context-window

## Time-Series
