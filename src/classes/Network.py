import numpy as np
import random
from utils.math import sigmoid, sigmoid_prime
from utils.lists import lzip

class Network(object):
  
  def __init__(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(y,1) for y in sizes[1:]]
    self.weights = [np.random.randn(y,x) for x,y in lzip(sizes[:-1], sizes[1:])]
    
  def feedforward(self, a):
    """Return the output of the network if "a" is input"""
    for b, w in lzip(self.biases, self.weights):
      a = sigmoid(np.dot(w, a) + b)
    return a
  
  def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    """Train the neural network employing Stochastic Gradient Descent, tuning
    a set of neurons instead of the whole network, greatly increasing performance
    at the minimal cost of short-term accuracy. 
    
    `training_data` is a list of tuples (x, y) representing the training inputs
    and the desired outputs, respectively."""
    # Set variables
    if test_data: n_test = len(test_data)
    n = len(training_data)
    
    # For each epoch
    for j in range(epochs):
      # shuffle training data and set mini_batches
      random.shuffle(training_data)
      mini_batches = [
        training_data[k:k+mini_batch_size]
        for k in range(0, n, mini_batch_size)
      ]
      
      # update all mini batches
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, eta)
        
      if test_data:
        print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
      else:
        print(f"Epoch {j} complete")
  
  def update_mini_batch(self, mini_batch, eta):
    # Initiate nabla_b and nabla_w with zeros
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    
    # Iterate through samples in the mini-batch
    for x, y in mini_batch:
      # set delta_nabla_b and delta_nabla_w with backprop
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)
      
      # update nabla_b and nabla_w with gradients from the input
      nabla_b = [nb+dnb for nb, dnb in lzip(nabla_b, delta_nabla_b)]
      nabla_w = [nw+dnw for nw, dnw in lzip(nabla_w, delta_nabla_w)]
      
    # after updating, update the weights and biases using the accumulated
    # gradients in nabla_b and nabla_w
    self.weights = [w-(eta/len(mini_batch))*nw 
                    for w, nw in lzip(self.weights, nabla_w)]
    self.biases  = [b-(eta/len(mini_batch))*nb
                    for b, nb in lzip(self.biases, nabla_b)]
    
  def backprop(self, x, y):
    """Returns a tuple `(nabla_b, nabla_w)` representing the gradient for the
    cost function C_x. `nabla_b` and `nabla_w` are layer by layer lists of numpy
    arrays."""
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    
    activation = x
    activations = [x] # list to store activations
    zs = [] # list to store all z vectors
    
    for b, w in lzip(self.biases, self.weights):
      z = np.dot(w, activation)+b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)
      
    # backward pass
    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    
    for l in range(2, self.num_layers):
      z = zs[-l]
      sp = sigmoid_prime(z)
      delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)
    
  def evaluate(self, test_data):
    """Return the number of tests the neural network outputs the correct result.
    Specifically made for the `MNIST number recognition problem`, since the index
    of the output neuron in the last layer is taken as the answer"""
    test_results = [(np.argmax(self.feedforward(x)), y)
                    for x,y in test_data]
    return sum(int(x == y) for x, y in test_results)
  
  def cost_derivative(self, output_activations, y):
    return (output_activations-y)