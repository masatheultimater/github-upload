from TLNN import *
from gradient import numerical_gradient
import numpy as np

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

x = np.random.rand(100, 784)
t = np.random.rand(100, 10)

grads = net.numerical_gradient(x, t)

print(grads['W1'].shape)
grads['b1'].shape
grads['W2'].shape
grads['b2'].shape
