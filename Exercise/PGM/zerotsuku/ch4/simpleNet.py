import sys, os
sys.path.append(os.pardir)
import numpy as np
import function

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = function.softmax(z)
        loss = function.cross_entropy_error(y, t)
        return loss
