import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    X = np.array(dict[b'data'])
    X = X.reshape((X.shape[0],3,32,32))
    Y = np.array(dict[b'labels'])
    return X,Y

def load_train_file():
    N, ch, dx, dy = 10000, 3, 32, 32
    n = 5

    X = np.empty((N * n, ch, dy, dx), dtype = np.int32)
    Y = np.empty(N * n, dtype = np.int32)

    for i in range(n):
        l = i * N
        r = l + N
        x, y = unpickle('./cifar-10-batches-py/data_batch_' + str(i + 1))
        X[l:r] = x
        Y[l:r] = y

    return X, Y

def load_test_file():
    X, Y = unpickle('./cifar-10-batches-py/test_batch')
    return X, Y

# プーリング幅
d2 = 2

# 畳み込み層
ch = 3
K = 3
R = 3
p = R // 2
s = 1

dx = dy = 32

dh = (2 * p + dy - R) // s + 1
dw = (2 * p + dx - R) // s + 1

# 各層の次元数
d = K * dh * dw // (d2 * d2)
M = 128
C = 10

# パラメータ保存用ファイル
# parameters_file = 'parameter.npz'
parameters_file = 'K3M128E100cifar.npz'

label = ['airplane', 'automobile', 'bird', 'cat' , 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def pre_process(X, Y):
    # (N, dy, dx) -> (N, 1, dy, dx)
    X = X / 255
    Y = np.array(list(map(one_hot_vector, Y)))
    return X, Y

def one_hot_vector(i):
    i = int(i)
    y = np.zeros(C)
    y[i] = 1
    return y

def post_process(y):
    i = np.argmax(y)
    return i
