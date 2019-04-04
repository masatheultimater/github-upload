#ANDパーセプトロン
#def AND(x1, x2):
#    w1, w2, theta = 0.5, 0.5, 0.7
#    tmp = x1*w1 + x2*w2
#    if tmp <= theta:
#        return 0
#    elif tmp > theta:
#        return 1
import numpy as np
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#NANDパーセプトロン
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#ORパーセプトロン
def OR(x1,x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#XORパーセプトロン
# s1: NAND
# s2: OR
# y : AND
# x1 x2 s1 s2 y
# 0  0  1  0  0
# 1  0  1  1  1
# 0  1  1  1  1
# 1  1  0  1  0

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print("***AND結果***")
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))
print("***NAND結果***")
print(NAND(0, 0))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(1, 1))
print("***OR結果***")
print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(1, 1))
print("***XOR結果***")
print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))