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

    X = np.empty((N, ch, dy, dx))
    Y = np.empty(N)

    for i in range(n):
        l = i * N
        r = l + N
        x, y = unpickle('./cifar-10-batches-py/data_batch_' + str(i + 1))
        X[l:r] = x
        Y[l:r] = y

    return X, Y
