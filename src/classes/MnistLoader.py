import pickle
import gzip
import numpy as np
from utils.lists import lzip

class MnistLoader:
  def __init__(self):
    self.training_data, self.validation_data, self.test_data = \
      self.load_data_wrapper()
  
  def load_data(self):
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
      training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)
  
  def load_data_wrapper(self):
    tr_d, va_d, te_d = self.load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [self.vectorized_result(y) for y in tr_d[1]]
    training_data = lzip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = lzip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = lzip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

  def vectorized_result(self, j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e