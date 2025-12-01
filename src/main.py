import numpy as np
from random import shuffle

def sigmoid(z):
  return 1.0/(1.0 + np.exp(-z))

class Network(object):
  
  def __init__(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(y,1) for y in sizes[1:]]
    self.weigths = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
    
  def feedforward(self, a):
    """Return the output of the network if "a" is input"""
    for b, w in zip(self.biases, self.weights):
      a = sigmoid(np.dot(w, a) + b)
      
    return a
  
  def SGC(self, training_data, epochs, mini_batch_size, eta, test_data=None):
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
      shuffle(training_data)
      mini_batches = [
        training_data[k:k+mini_batches]
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
      nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
      
    # after updating, update the weights and biases using the accumulated
    # gradients in nabla_b and nabla_w
    self.weigths = [w-(eta/len(mini_batch))*nw 
                    for w, nw in zip(self.weights, nabla_w)]
    self.biases  = [b-(eta/len(mini_batch))*nabla_b
                    for b, nb in zip(self.biases, nabla_b)]
    
  
  
net = Network([2, 3, 1])