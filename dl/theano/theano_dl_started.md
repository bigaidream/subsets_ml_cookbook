<!-- toc -->

# Deep Learning with Theano, Getting Started

> http://deeplearning.net/tutorial/gettingstarted.html#gettingstarted

## MNIST dataset
```python
import cPickle, gzip
f = gzip.open('/home/jie/.keras/datasets/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
```
For GPU, we use `shared` variables. By doing so, theano can copy the entire data on the GPU in a single call. 

Because labels are integers while raw features are usually real numbers, it's recommended to use different variables for labels and data. Also it's recommended using different variables for the training set, validation set and testing set. 

Since now the data is in one variable, and a minibatch is defined as a slice of that variable, it is natural to define a minibatch by indicating its index and its size. 

```python
def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')

test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)

batch_size = 500    # size of the minibatch

# accessing the third minibatch of the training set

data  = train_set_x[2 * batch_size: 3 * batch_size]
label = train_set_y[2 * batch_size: 3 * batch_size]
```

### Learning a classifier
#### Zero-One Loss

```python
zero_one_loss = T.sum(T.neq(T.argmax(p_y_given_x), y))
```
#### Negative Log-Likelihood Loss
Since the zero-one loss is not differentiable, optimizing it for large models (thousands or millions of parameters) is prohibitively expensive (computationally). We thus usually maximize the log-likelihood of our classifier given all the labels in a training set.



The `NLL` of our classifier is a differentiable **surrogate** for the `zero-one loss`. The code is:
```python
NLL = -T.sum(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
# note on syntax: T.arange(y.shape[0]) is a vector of integers [0,1,2,...,len(y)].
# Indexing a matrix M by the two vectors [0,1,...,K], [a,b,...,k] returns the
# elements M[0,a], M[1,b], ..., M[K,k] as a vector.  Here, we use this
# syntax to retrieve the log-probability of the correct labels, y.
```

### Stochastic Gradient Descent
```python
# Minibatch SGD

# assume loss is a symbolic description of the loss function given
# the symbolic variables params (shared variable), x_batch, y_batch;

# compute gradient of loss w.r.t. params
d_loss_wrt_params = T.grad(loss, params)

# compile the MSGD step into a theano function
updates = [(params, params - learning_rate * d_loss_wrt_params)]
MSGD = theano.function([x_batch,y_batch], loss, updates=updates)

for (x_batch, y_batch) in train_batches:
    # here x_batch and y_batch are elements of train_batches and
    # therefore numpy arrays; function MSGD also updates the params
    print('Current loss is ', MSGD(x_batch, y_batch))
    if stopping_condition_is_met:
        return params
```

### Regularization
#### L1 and L2 regularization
L1 and L2 regularization involve adding an extra term to the loss function, which penalizes certain parameter configurations.

The regularized loss will be:

$$E(\theta, \mathcal{D}) =  NLL(\theta, \mathcal{D}) + \lambda R(\theta)\\$$

Or more specifically, in our case 

$$E(\theta, \mathcal{D}) =  NLL(\theta, \mathcal{D}) + \lambda||\theta||_p^p$$

where

$$||\theta||_p = \left(\sum_{j=0}^{|\theta|}{|\theta_j|^p}\right)^{\frac{1}{p}}$$

which is the $$L_p$$ norm of $$\theta$$. 

intuitively, the two terms (NLL and $$R(\theta)$$)
correspond to modelling the data well (NLL) and having "simple" or "smooth" solutions ($$R(\theta)$$). Thus, minimizing the sum of both will, in theory, correspond to finding the right trade-off between the fit to the training data and the "generality" of the solution that is found. To follow
Occam's razor principle, this minimization should find us the simplest solution (as measured by our simplicity criterion) that fits the training data.

Note that the fact that a solution is “simple” does not mean that it will generalize well. Empirically, it was found that performing such regularization in the context of neural networks helps with generalization, especially on `small`datasets. 

The code block below shows how to compute the loss in python when it contains both a L1 regularization term weighted by $$\lambda_1$$ and L2 regularization term weighted by $$\lambda_2$$

```python
# symbolic Theano variable that represents the L1 regularization term
L1  = T.sum(abs(param))

# symbolic Theano variable that represents the squared L2 term
L2_sqr = T.sum(param ** 2)

# the loss
loss = NLL + lambda_1 * L1 + lambda_2 * L2
```

#### Early stopping
Early-stopping combats overfitting by monitoring the model's performance on a `validation set`. A validation set is a set of examples that we never use for gradient descent, but which is also not a part of the test set. The validation examples are considered to be representative of future test examples. We can use them during training because they are not part of the test set. If the model’s performance ceases to improve sufficiently on the validation set, or even degrades with further optimization, then the heuristic implemented here gives up on much further optimization.

The choice of when to stop is a judgement call and a few heuristics exist, but these tutorials will make use of a strategy based on a geometrically increasing amount of patience.

```python
# early-stopping parameters
patience = 5000  # look as this many examples regardless
patience_increase = 2     # wait this much longer when a new best is
                              # found
improvement_threshold = 0.995  # a relative improvement of this much is
                               # considered significant
validation_frequency = min(n_train_batches, patience/2)
                              # go through this many
                              # minibatches before checking the network
                              # on the validation set; in this case we
                              # check every epoch

best_params = None
best_validation_loss = numpy.inf
test_score = 0.
start_time = time.clock()

done_looping = False
epoch = 0
while (epoch < n_epochs) and (not done_looping):
    # Report "1" for first epoch, "n_epochs" for last epoch
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):

        d_loss_wrt_params = ... # compute gradient
        params -= learning_rate * d_loss_wrt_params # gradient descent

        # iteration number. We want it to start at 0.
        iter = (epoch - 1) * n_train_batches + minibatch_index
        # note that if we do `iter % validation_frequency` it will be
        # true for iter = 0 which we do not want. We want it true for
        # iter = validation_frequency - 1.
        if (iter + 1) % validation_frequency == 0:

            this_validation_loss = ... # compute zero-one loss on validation set

            if this_validation_loss < best_validation_loss:

                # improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss * improvement_threshold:

                    patience = max(patience, iter * patience_increase)
                best_params = copy.deepcopy(params)
                best_validation_loss = this_validation_loss

        if patience <= iter:
            done_looping = True
            break

# POSTCONDITION:
# best_params refers to the best out-of-sample parameters observed during the optimization
```

> The `validation_frequency` should always be smaller than the patience. The code should check at least two times how it performs before running out of patience. This is the reason we used the formulation `validation_frequency = min( value, patience/2.)`

> This algorithm could possibly be improved by using a test of statistical significance rather than the simple comparison, when deciding whether to increase the patience.

### Testing
When we have finally chosen the model we think is the best (on validation data), we report that model’s test set performance.

## Theano/Python Tips
### Loading and Saving Models
Gradient-descent learning is slow. We will want to save those weights once you find them. We may also want to save the current-best estimates as the search progresses.

### Pickle the numpy ndarrays from your shared variables
The best way to save/archive your model’s parameters is to use pickle or deepcopy the ndarray objects. So for example, if your parameters are in shared variables `w`, `v`, `u`, then your save command should look something like:
```python
import cPickle
save_file = open('path', 'wb')  # this will overwrite current contents
cPickle.dump(w.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
cPickle.dump(v.get_value(borrow=True), save_file, -1)  # .. and it triggers much more efficient
cPickle.dump(u.get_value(borrow=True), save_file, -1)  # .. storage than numpy's default
save_file.close()
```
Then later, you can load your data back like this:
```python
save_file = open('path')
w.set_value(cPickle.load(save_file), borrow=True)
v.set_value(cPickle.load(save_file), borrow=True)
u.set_value(cPickle.load(save_file), borrow=True)
```

### Do not pickle your training or test functions for long-term storage