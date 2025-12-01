from classes.Network import Network
from classes.MnistLoader import MnistLoader

mnist = MnistLoader()

net = Network([784, 30, 10])

net.SGD(mnist.training_data, 30, 10, 3.0, test_data=mnist.test_data)