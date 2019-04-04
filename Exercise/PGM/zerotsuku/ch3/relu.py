import numpy as np
import matplotlib.pyplot as plt

#ReLu
def relu(x):
    return np.maximum(x, 0)

x = np.arange(-6, 6, 0.1)
y = relu(x)
plt.plot(x,y)
plt.ylim(-1, 5.1)
plt.show()
