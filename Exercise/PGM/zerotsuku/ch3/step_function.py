import numpy as np
import matplotlib.pylab as plt

#ステップ関数

#引数が実数のステップ関数
#def step_function(x):
#    if x >= 0:
#        return 1
#    else:
#        return 0

#引数が行列のステップ関数
def step_function(x):
#    y = x > 0
#    return y.astype(np.int)
    return np.array(x > 0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()
