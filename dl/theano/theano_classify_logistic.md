<!-- toc -->

# Theano for Logistic Regression

> http://deeplearning.net/tutorial/logreg.html

## The model
Classification is done by projecting an input vector onto a set of hyperplanes, each of which corresponds to a class. The distance from the input to a hyperplane reflects the probability that the input is a member of the corresponding class.

Mathematically, the probability that an input vector $$x$$ is a member of a class $$i$$, a value of a stochastic variable $$Y$$, can be written as:

$$P(Y=i|x,W,b)=softmax_{i}(Wx+b)$$

$$=\frac{e^{W_{i}x+b_{i}}}{\sum_{j}e^{W_{j}x+b_{j}}}$$

The model's prediction $$y_{pred}$$ is the class whose probability is maximal:

$$y_{pred} = {\rm argmax}_i P(Y=i|x,W,b)$$

The code in Theano is:
```python
class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
```

> The result `p_y_given_x` is a symbolic variable of `matrix`-type, rather than `vector`-type in the original webpage. We can see that from the use of `axis=1`

## Defining a loss function

Learning optimal model parameters involves minimizing a loss function. In the case of multi-class logistic regression, it is very common to use the negative log-likelihood as the loss. This is equivalent to maximizing the likelihood of the data set $$\cal{D}$$ under the model parameterized by $$\theta$$. Let us first start by defining the likelihood $$\cal{L}$$ and loss $$\ell$$:

$$\mathcal{L}(\theta=\{W,b\},\mathcal{D})=\sum_{i=0}^{|\mathcal{D}|}\log(P(Y=y^{(i)}|x^{(i)},W,b))$$

$$\ell(\theta=\{W,b\},\mathcal{D})=-\mathcal{L}(\theta=\{W,b\},\mathcal{D})$$

The following Theano code defines the (symbolic) loss for a given minibatch:
```python
# y.shape[0] is (symbolically) the number of rows in y, i.e.,
# number of examples (call it n) in the minibatch
# T.arange(y.shape[0]) is a symbolic vector which will contain
# [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
# Log-Probabilities (call it LP) with one row per example and
# one column per class LP[T.arange(y.shape[0]),y] is a vector
# v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
# LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
# the mean (across minibatch examples) of the elements in v,
# i.e., the mean log-likelihood across the minibatch.
return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
```

Even though the loss is formally defined as the sum, over the data set, of individual error terms, `in practice`, we use the `mean` (T.mean) in the code. This allows for the learning rate choice to be less dependent of the minibatch size.

## Creating a logisticRegresion class

> http://deeplearning.net/tutorial/code/logistic_sgd.py

This class can be instantiated as follows:
```python
# generate symbolic variables for input (x and y represent a
# minibatch)
x = T.matrix('x')  # data, presented as rasterized images
y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

# construct the logistic regression class
# Each MNIST image has size 28*28
classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
```

Finally, we define a (symbolic) `cost` variable to minimize, using the instance method `classifier.negative_log_likelihood`.

```python
# the cost we minimize during training is the negative log likelihood of
# the model in symbolic format
cost = classifier.negative_log_likelihood(y)
```

Note that `x` is an implicit symbolic input to the definition of `cost`, because the symbolic variables of `classifier` were defined in terms of x at initialization.

## Learning the model

To get the gradients $$\partial{\ell}/\partial{W}$$ and
$$\partial{\ell}/\partial{b}$$ in Theano, simply do the following:
```python
g_W = T.grad(cost=cost, wrt=classifier.W)
g_b = T.grad(cost=cost, wrt=classifier.b)
```
`g_W` and `g_b` are symbolic variables, which can be used as part of a computation graph. The function train_model, which performs one step of gradient descent, can then be defined as follows:
```python
# specify how to update the parameters of the model as a list of
# (variable, update expression) pairs.
updates = [(classifier.W, classifier.W - learning_rate * g_W),
           (classifier.b, classifier.b - learning_rate * g_b)]

# compiling a Theano function `train_model` that returns the cost, but in
# the same time updates the parameter of the model based on the rules
# defined in `updates`
train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)
```

`updates` is a list of pairs. In each pair, the first element is the symbolic variable to be updated in the step, and the second element is the symbolic function for calculating its new value. Similarly, `givens` is a dictionary whose keys are symbolic variables and whose values specify their replacements during the step. The function `train_model` is then defined such that:
1. the input is the mini-batch index `index` that, together with the `batch_size` (which is **NOT** an input since it is **fixed**) defines $$x$$ with corresponding labels $$y$$
2. the return value is the cost/loss associated with the x,y defined by the `index`
3. on every function call, it will first replace `x` and `y` with the slices from the training set specified by `index`. Then, it will evaluate the cost associated with that minibatch and apply the operations defined by the `updates` list. 

Each time `train_model(index)` is called, it will thus compute and return the cost of a minibatch, while also performing a step of MSGD. The entire learning algorithm thus consists in looping over all examples in the dataset, considering all the examples in one minibatch at a time, and repeatedly calling the `train_model` function. 

## Testing the model
When testing, we care about the nuber of misclassified examples. `LogisticRegression` has `errors()` method for retrieving the number of misclassified examples in each minibatch. 

`validate_model` is key to early-stopping implementation. Both `test_model` and `validate_model` take a minibatch index and compute, for the examples in that minibatch, the number that were misclassified by the model. 