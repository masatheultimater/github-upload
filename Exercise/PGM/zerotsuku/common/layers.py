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

class Relu:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None #損失
        self.y = None #softmaxの出力
        self.t = None #教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
