# Gaussian Processes

## Current Understandings

GPs combine the flexibility of being capable of modelling arbitrary smooth functions if given enough data, with the simplicity of a Bayesian specification that only requires inference over a small number of readily interpretable hyperparameters (in contrast to deep neural networks) -- such as the length-scales by which the function varies along different dimensions, the contributions of signal and noise to the variance in the data, etc [*Infinite Mixtures of Gaussian Process Experts*, NIPS 2012]. 

As a probabilistic non-parametric framework, it can model middle-scale data very well. Of course, we can use a varieties of sparse (Neial Lawrence once argued that we should call it `low-rank`) approximation methods to enable it to handle millions of datapoints. But in that case, it does not seems capable of competing with deep neural networks. 

