import mnist
import numpy as np

# シードの設定
np.random.seed(100)

# 読み込み関数のエイリアス
load_file = mnist.download_and_parse_mnist_file

# データファイル
train_images_file = 'train-images-idx3-ubyte.gz'
train_labels_file = 'train-labels-idx1-ubyte.gz'
test_images_file = 't10k-images-idx3-ubyte.gz'
test_labels_file = 't10k-labels-idx1-ubyte.gz'

# プーリング幅
d2 = 2

# 畳み込み層
ch = 1
K = 4
R = 3
p = R // 2
s = 1

dx = dy = 28

dh = (2 * p + dy - R) // s + 1
dw = (2 * p + dx - R) // s + 1

# 各層の次元数
d = K * dh * dw // (d2 * d2)
M = 128
C = 10

# print(dh, dw, d)

# パラメータ保存用ファイル
parameters_file = 'parameter.npz'
# parameters_file = 'K16M512E20.npz'
# parameters_file = 'K4M256E30.npz'
# parameters_file = 'rl_do_bn_ad_cp_M128_E10.npz'

def pre_process(X, Y):
    # (N, dy, dx) -> (N, 1, dy, dx)
    X = X[:, np.newaxis, :, :] / 255
    Y = np.array(list(map(one_hot_vector, Y)))
    return X, Y

def one_hot_vector(i):
    y = np.zeros(C)
    y[i] = 1
    return y

def post_process(y):
    i = np.argmax(y)
    return i
