<!-- toc -->

# JAGS with R Tutorial

> https://github.com/ml-playground/kruschke-doing_bayesian_data_analysis/blob/master/2e/Jags-ExampleScript.R

> [Doing Bayesian Data Analysis], Chapter 8

A parameterized model consists of a likelihood function, which specifies the probability of data given the parameter values, and a prior distribution, which specifies the probability of candidate parameter values without taking into account the data. 

## Load data

Logically, models of data start with the data. 

To bundle the data for JAGS, we put it into a `list` structure. 

https://github.com/ml-playground/kruschke-doing_bayesian_data_analysis/blob/master/2e/Jags-ExampleScript.R#L16-L23

## Specify model
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

## Initialize chains

In general, a useful choice for initial values of the parameters is their maximum likelihood estimate (MLE). The MLE is the value of the parameter that maximizes the likelihood function, which is the value of the parameter that maximizes the probability of the data. 