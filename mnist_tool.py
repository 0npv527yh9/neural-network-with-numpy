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

# 入力画像の次元数 dy * dx
dx = dy = 28

# 畳み込み層
ch = 1
K = 4
R = 3
p = R // 2
s = 1

# プーリング幅
d2 = 2

#　畳み込み後の次元数 dw * dh
dh = (2 * p + dy - R) // s + 1
dw = (2 * p + dx - R) // s + 1

# 全結合層の次元数
d = K * dh * dw // (d2 * d2)
M = 128
C = 10

# パラメータ保存用ファイル
parameters_file = 'mnist_para.npz'

# 前処理
def pre_process(X):
    # (N, 1, dy, dx)
    X = X[:, np.newaxis, :, :] / 255
    return X

# 後処理
def post_process(y):
    i = np.argmax(y)
    return i
