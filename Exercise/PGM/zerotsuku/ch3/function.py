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
