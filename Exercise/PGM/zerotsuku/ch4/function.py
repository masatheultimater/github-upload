import numpy as np

#ReLu
def relu(x):
    return np.maximum(x, 0)
#シグモイド関数
def sigmoid(x):
    return 1/ (1 + np.exp(-x))
#引数が行列のステップ関数
def step_function(x):
#    y = x > 0
#    return y.astype(np.int)
    return np.array(x > 0, dtype=int)
#ソフトマックス
def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp = np.sum(exp_x)
    y = exp_x / sum_exp
    return y
#最小二乗法
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)
#交差エントロピー
#def cross_entropy_error(y, t):
#    delta = 1e-7 #log(0)時のエラー回避
#    return -np.sum(t * np.log(y + delta))
#交差エントロピー(ミニバッチ)
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    delta = 1e-7 #log(0)時のエラー回避
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
