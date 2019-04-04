import numpy as np
import matplotlib.pyplot as plt
#シグモイド関数
def sigmoid(x):
    return 1/ (1 + np.exp(-x))

#スカラ値と行列の積はブロードキャストされ，各要素ごとに計算される
#x = np.array([-1, 1, 2])
#print(sigmoid(x))

#表示
#x = np.arange(-5, 5, 0.1)
#y = sigmoid(x)
#plt.plot(x,y)
#plt.ylim(-0.1, 1.1)
#plt.show()
