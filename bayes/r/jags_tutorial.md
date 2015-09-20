<!-- toc -->

> start: 2015-09-20

# JAGS with R Tutorial

> https://github.com/ml-playground/kruschke-doing_bayesian_data_analysis/blob/master/2e/Jags-ExampleScript.R

> [Doing Bayesian Data Analysis], Chapter 8

A parameterized model consists of a likelihood function, which specifies the probability of data given the parameter values, and a prior distribution, which specifies the probability of candidate parameter values without taking into account the data. 

## A complete example

### Load data

Logically, models of data start with the data. 

To bundle the data for JAGS, we put it into a `list` structure. 

https://github.com/ml-playground/kruschke-doing_bayesian_data_analysis/blob/master/2e/Jags-ExampleScript.R#L16-L23

### Specify model
```r
model {
  for ( i in 1:Ntotal) {
    y[i] ~ dbern( theta)
  }
  theta ~ dbeta(1, 1)
}
```

JAGS itself does not care about the ordering of the statements, because JAGS does not execute the model statement as an ordered procedure. 

To get the model specification into JAGS, we need to create the specification as a character string to R (similar to the situations in Neon, where they use `.yaml`). 

### Initialize chains

In general, a useful choice for initial values of the parameters is their maximum likelihood estimate (MLE). The MLE is the value of the parameter that maximizes the likelihood function, which is the value of the parameter that maximizes the probability of the data. 

A recommended approach is to start the chains at random points near the MLE. 

---

A trick to keep `thetaInit` away from 0, 1:
```r
thetaInit = 0.001 + 0.998*thetaInit
```

### Generate chains
First, we do some burn-in using `update` function:
```r
update( jagsModel, n.iter=500)
```
The `update` function returns no values, and it merely changes the internal state of the `jagsModel` object (similar to shared variables in Theano). It does *not* record the sampled parameter values during the updating. 

### Examine chains

> `plotPost` is in https://github.com/ml-playground/kruschke-doing_bayesian_data_analysis/blob/master/2e/DBDA2E-utilities.R#L355

## Sampling from the prior distribution in JAGS

It is straight forward to have JAGS generate an MCMC sample from the prior: we simply run the program with no data included. 

To run JAGS without the data included, we must omit the y values, but we must retain all constants that define the *structure* of the model. 

## Defining new likelihood functions
It is important to realize that what we would get from `y[i] ~ pdf(parameters)` is the value of the pdf when `y[i]` has its particular value and when the parameters have their particular randomly generated MCMC values. 

## Faster sampling with parallel processing in runjags

The essential difference between running JAGS via `rjags` and `runjags` is merely the functions used to invoke JAGS. A key advantage of `runjags` is specified by the first argument in the `run.jags` command above, named `method`. This is where we can tell `rungas` to run parallel chains on multiple cores. 