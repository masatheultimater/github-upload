import numpy as np

#数値微分
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h) / (2*h))

#数値微分を使用した勾配計算
#def numerical_gradient(f, x):
#    h = 1e-4
#    grad = np.zeros_like(x)
#
#    for idx in range(x.size):
#        tmp_val = x[idx]
#        x[idx] = tmp_val + h
#        fxh1 = f(x)
#        x[idx] = tmp_val - h
#        fxh2 = f(x)
#        grad[idx] = (fxh1 - fxh2) / (2*h)
#        x[idx] = tmp_val
#    return grad
#
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 値を元に戻す
        it.iternext()

    return grad

#勾配降下法(数値微分を使用)
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x
