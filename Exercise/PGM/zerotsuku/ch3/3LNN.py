#3Layer NN 3 * 3 * 2 * 2
import numpy as np
from sigmoid import sigmoid

#パラメータ初期化
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

#行列の確認
print(W1.shape)
print(X.shape)
print(B1.shape)

#中間1層での総入力
A1 = np.dot(X, W1) + B1
print("A1での総入力:" + str(A1))

#中間1層での活力化関数
Z1 = sigmoid(A1)
print("Z1での出力:" + str(Z1))

#行列の確認
print(Z1.shape)
print(W2.shape)
print(B2.shape)

#中間2層での総入力
A2 = np.dot(Z1, W2) + B2
print("A2での総入力:" + str(A2))

#中間2層での活力化関数
Z2 = sigmoid(A2)
print("Z2での出力:" + str(Z2))

#出力層 活性化関数 恒等写像
def identity_function(x):
    return x

#出力層での総入力
A3 = np.dot(Z2, W3) + B3
print("A3での総入力:" + str(Z2))

#出力層での活性化関数
Y = identity_function(A3)
print("総出力:" + str(Y))
