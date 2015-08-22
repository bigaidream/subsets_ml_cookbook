<!-- toc -->

# Hyperopt with Sklearn

## Issue with `hpsklearn.components.any_processing`

The example given in https://github.com/hyperopt/hyperopt-sklearn/blob/master/notebooks/Demo-Iris.ipynb does not work due to the `pca` component. I have either to a) manually set the `n_comnents=actual_input_dim` b) disable `pca`. 
