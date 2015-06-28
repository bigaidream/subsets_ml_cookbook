<!-- toc -->
# Book: Neural Networks and Deep Learning

I'm using this material to refresh my deep learning knowledge. 

## Basic Network for MNIST

> http://neuralnetworksanddeeplearning.com/chap1.html#perceptrons

```python
class Network():
	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		# this assignment is kinda clever
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	def feedforward(self, a):
		for b, w in zip(self.biases, self.weights):
			a = sigmoid_vec(np.dot(w, a) + b)
		return a
	
	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
		"""Train the neural network using mini-batch sgd.
		Parameters
		----------
		training_data :	a list of tuples ``(x, y)``, shape = [n_samples, n_features]
		The training inputs and the desired outputs.
		"""
```
We need to use `np.vectorize` to define `sigmoid` as follows:
```python
def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))
sigmoid_vec = np.vectorize(sigmoid)
```


