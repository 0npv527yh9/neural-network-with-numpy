import numpy as np

# 全結合層
class Affine:
    def __init__(self):
        self.adam_W = Adam()
        self.adam_b = Adam()

    def init(self, row, col):
        self.W = random_ndarray(col, (row, col))
        self.b = random_ndarray(col, row)

    def save(self, para, name):
        para[name + '_W'] = self.W
        para[name + '_b'] = self.b
        self.adam_W.save(para, name + '_adam_W')
        self.adam_b.save(para, name + '_adam_b')

    def load(self, para, name):
        self.W = para[name + '_W']
        self.b = para[name + '_b']
        self.adam_W.load(para, name + '_adam_W')
        self.adam_b.load(para, name + '_adam_b')

    def forward(self, x):
        self.x = x
        self.y = (np.dot(self.W, x).T + self.b).T
        return self.y

    def backward(self, dEn_dY):
        dEn_dX = np.dot(self.W.T, dEn_dY)
        dEn_dW = np.dot(dEn_dY, self.x.T)
        dEn_db = dEn_dY.sum(axis = 1)
        self.W -= self.adam_W.grad(dEn_dW)
        self.b -= self.adam_b.grad(dEn_db)
        return dEn_dX

class Sigmoid:
    def forward(self, t):
        self.x = t
        self.y = 1 / (1 + np.exp(-t))
        return self.y

    def backward(self, dEn_dy):
        return dEn_dy * (1 - self.y) * self.y

class Softmax:
    def forward(self, a):
        self.x = a
        ex = np.exp(a - np.max(a, axis = 0))
        self.y = ex / np.sum(ex, axis = 0)
        return self.y

    def backward(self, y, B):
        cols = np.arange(B)
        self.y[y, cols] -= 1
        self.y /= B
        return self.y
        # return (self.y - y) / B

class ReLU:
    def forward(self, t):
        self.x = t
        return np.maximum(0, t)

    def backward(self, dEn_dy):
        return np.where(self.x > 0, dEn_dy, 0)

class Dropout:
    def forward(self, x, is_train, rate = 0.5):
        if is_train:
            self.is_valid = np.random.rand(*x.shape) > rate
            y = np.where(self.is_valid, x, 0)
        else:
            y = x * (1 - rate)
        return y

    def backward(self, dEn_dy):
        return np.where(self.is_valid, dEn_dy, 0)

class Batch_normalization:
    eps = 1e-10
    def __init__(self):
        self.adam_gamma = Adam()
        self.adam_beta = Adam()
        self.gamma = 1
        self.beta = 0
        self.var_sum = 0.
        self.mean_sum = 0.
        self.count = 0

    def save(self, para, name):
        para[name + '_gamma'] = self.gamma
        para[name + '_beta'] = self.beta
        para[name + '_var_sum'] = self.var_sum
        para[name + '_mean_sum'] = self.mean_sum
        para[name + '_count'] = self.count
        self.adam_gamma.save(para, name + '_adam_gamma')
        self.adam_beta.save(para, name + '_adam_beta')

    def load(self, para, name):
        self.gamma  = para[name + '_gamma']
        self.beta = para[name + '_beta']
        self.var_sum = para[name + '_var_sum']
        self.mean_sum = para[name + '_mean_sum']
        self.count = para[name + '_count']
        self.adam_gamma.load(para, name + '_adam_gamma')
        self.adam_beta.load(para, name + '_adam_beta')

    def forward(self, x, is_train):
        self.x = x
        if is_train:
            self.mean = np.mean(x, axis = 1)
            self.var = np.var(x, axis = 1)
            self.x_hat = ((x.T - self.mean) / np.sqrt(self.var + Batch_normalization.eps)).T
            self.y = ((self.gamma * self.x_hat.T) + self.beta).T

            # B = x.shape[1]
            # self.var_sum += self.var * B / (B - 1)
            # self.mean_sum += self.mean

            momentum = 0.01
            self.mean_sum *= (1 - momentum)
            self.mean_sum += momentum * self.mean
            self.var_sum *= (1 - momentum)
            self.var_sum += momentum * self.var

        else:
            var = self.var_sum
            mean = self.mean_sum
            # var = self.var_sum / self.count if self.count > 0 else 0
            # mean = self.mean_sum / self.count if self.count > 0 else 0
            c = self.gamma / np.sqrt(var + Batch_normalization.eps)
            self.y = (c * x.T + (self.beta - c * mean)).T
        return self.y

    def backward(self, dEn_dy):
        self.count += 1

        B = dEn_dy.shape[1]
        dEn_dx_hat = (dEn_dy.T * self.gamma).T
        dEn_dvar = np.sum(dEn_dx_hat * (self.x.T - self.mean).T, axis = 1) * (-1 / 2) * (self.var + Batch_normalization.eps) ** (-3 / 2)
        c = 1 / np.sqrt(self.var + Batch_normalization.eps)
        dEn_dmean = -c * np.sum(dEn_dx_hat, axis = 1) + dEn_dvar * (-2) * (np.mean(self.x, axis = 1) - self.mean)
        dEn_dx = (dEn_dx_hat.T * c + dEn_dvar * 2 * (self.x.T - self.mean) / B + dEn_dmean / B).T

        dEn_dgamma = np.sum(dEn_dy * self.x_hat, axis = 1)
        dEn_dbeta = np.sum(dEn_dy, axis = 1)
        self.gamma -= self.adam_gamma.grad(dEn_dgamma)
        self.beta -= self.adam_beta.grad(dEn_dbeta)

        return dEn_dx

class Adam:
    def __init__(self):
        self.t = 0
        self.m = 0
        self.v = 0

    def save(self, para, name):
        para[name + '_t'] = self.t
        para[name + '_m'] = self.m
        para[name + '_v'] = self.v

    def load(self, para, name):
        self.t = para[name + '_t']
        self.m = para[name + '_m']
        self.v = para[name + '_v']

    def grad(self, dEn_dW, alpha = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        self.t += 1
        self.m *= beta1
        self.m += (1 - beta1) * dEn_dW
        self.v *= beta2
        self.v += (1 - beta2) * dEn_dW * dEn_dW
        m_hat = self.m / (1 - np.power(beta1, self.t))
        v_hat = self.v / (1 - np.power(beta2, self.t))
        return alpha * m_hat / (np.sqrt(v_hat) + eps)

# 畳み込み層
class Convolution:
    def __init__(self):
        self.adam_W = Adam()
        self.adam_b = Adam()

    def init(self, ch, K, R, p = -1, s = 1):
        self.R = R
        self.p = R // 2 if p < 0 else p
        self.s = s

        row = K
        col = R * R * ch
        self.W = random_ndarray(col, (row, col))
        self.b = random_ndarray(col, row)

    def save(self, para, name):
        para[name + '_R'] = self.R
        para[name + '_p'] = self.p
        para[name + '_s'] = self.s

        para[name + '_W'] = self.W
        para[name + '_b'] = self.b
        self.adam_W.save(para, name + '_adam_W')
        self.adam_b.save(para, name + '_adam_b')

    def load(self, para, name):
        self.R = para[name + '_R']
        self.p = para[name + '_p']
        self.s = para[name + '_s']

        self.W = para[name + '_W']
        self.b = para[name + '_b']
        self.adam_W.load(para, name + '_adam_W')
        self.adam_b.load(para, name + '_adam_b')

    def forward(self, x):
        # (R * R * ch, dw * dh * B)
        self.x, (B, dh, dw) = convert_batch_into_X(x, self.R, self.p, self.s)

        # 畳み込み演算
        y = (np.dot(self.W, self.x).T + self.b).T

        # (B, K, dh, dw)
        y = y.reshape(-1, B, dh, dw).transpose(1, 0, 2, 3)

        return y

    def backward(self, dEn_dY):
        B, K, dh, dw = dEn_dY.shape

        # (K, dw * dh * B)
        dEn_dY = dEn_dY.transpose(1, 0, 2, 3).reshape(K, -1)

        dEn_dX = np.dot(self.W.T, dEn_dY)
        dEn_dW = np.dot(dEn_dY, self.x.T)
        dEn_db = dEn_dY.sum(axis = 1)
        self.W -= self.adam_W.grad(dEn_dW)
        self.b -= self.adam_b.grad(dEn_db)

        # (B, ch, dy, dx)
        dEn_dX = convert_X_into_batch(dEn_dX, self.R, self.p, self.s, (B, dh, dw))
        return dEn_dX

# プーリング層
class Pooling:
    def init(self, d):
        self.d = d

    def save(self, para, name):
        para[name + '_d'] = self.d

    def load(self, para, name):
        self.d = para[name + '_d']

    def forward(self, x):
        self.x = x
        B, ch, dy, dx = x.shape

        dh = dy // self.d
        dw = dx // self.d

        # (d, d, B, ch, dh, dw)
        x = x.reshape(B, ch, dh, self.d, dw, self.d).transpose(3, 5, 0, 1, 2, 4)

        # (d * d, dw * dh * ch * B)
        x = x.reshape(self.d * self.d, -1)

        # dw * dh * ch * B
        y = x.max(axis = 0)

        #  (d * d, dw * dh * ch * B)
        self.mask = x == y

        # (B, ch, dh, dw)
        y = y.reshape(B, ch, dh, dw)

        return y

    def backward(self, dEn_dY):
        B, ch, dh, dw = dEn_dY.shape

        dy = dh * self.d
        dx = dw * self.d

        # dw *dh * ch * B
        dEn_dY = dEn_dY.reshape(-1)

        # (d * d, dw * dh * ch * B)
        dEn_dX = np.where(self.mask, dEn_dY, 0)

        # (B, ch, dy, dx)
        dEn_dX = dEn_dX.reshape(self.d, self.d, B, ch, dh, dw).transpose(2, 3, 4, 0, 5, 1).reshape(B, ch, dy, dx)
        return dEn_dX

    # (B, ch, dy, dx) -> (dx * dy * ch, B)
    def convert_images_into_vectors(self, X):
        self.shape = X.shape
        B = self.shape[0]
        return X.transpose(1, 2, 3, 0).reshape(-1, B)

    # (dx * dy * ch, B) -> (B, ch, dy, dx)
    def convert_vectors_into_images(self, Y):
        B, ch, dy, dx = self.shape
        return Y.reshape(ch, dy, dx, B).transpose(3, 0, 1, 2)

class Neural_network:
    # 初期化
    def init(self, d, M, C, ch, K, R, p, s, d2):
        self.layers = create_layers()
        self.layers['conv'].init(ch, K, R, p, s)
        self.layers['pooling'].init(d2)
        self.layers['affine1'].init(M, d)
        self.layers['affine2'].init(C, M)
        self.epoch_count = 0

    # パラメータ保存
    def save(self, file_name):
        para = {'epoch_count': self.epoch_count}

        # 各層で save を呼び出す
        for name, layer in self.layers.items():
            if hasattr(layer, 'save'):
                layer.save(para, name)

        np.savez(file_name, **para)

    # パラメータ読み込み
    def load(self, file_name):
        para = np.load(file_name)

        # 各層で load を呼び出す
        self.layers = create_layers()
        for name, layer in self.layers.items():
            if hasattr(layer, 'load'):
                layer.load(para, name)

        self.epoch_count = para['epoch_count']

    # 順伝播
    def forward(self, x, is_train = False):
        # 畳み込み
        x = self.layers['conv'].forward(x)

        # Batch Normalization
        B, ch, dh, dw = x.shape
        # (dw * dh * ch, B)
        x = x.transpose(1, 2, 3, 0).reshape(-1, B)
        x = self.layers['bn1'].forward(x, is_train)
        # (B, ch, dh, dw)
        x = x.reshape(ch, dh, dw, B).transpose(3, 0, 1, 2)

        # ReLu と Pooling
        x = self.layers['relu1'].forward(x)
        x = self.layers['pooling'].forward(x)
        # (d, B)
        x = self.layers['pooling'].convert_images_into_vectors(x)

        x = self.layers['affine1'].forward(x)
        x = self.layers['bn2'].forward(x, is_train)
        x = self.layers['relu2'].forward(x)
        x = self.layers['dropout'].forward(x, is_train)
        x = self.layers['affine2'].forward(x)
        x = self.layers['softmax'].forward(x)

        return x

    # ミニバッチ学習
    def mini_batch_training(self, X, Y, B):
        mini_batch = X, Y = create_mini_batch(X, Y, B)
        Y2 = self.forward(X, is_train = True)
        En = np.mean(cross_entropy(Y, Y2))
        return mini_batch, En

    # 誤差逆伝播
    def backward(self, X_train, Y_train, X_test, Y_test, B, epoch):
        # 訓練データの総数
        N = len(X_train)
        #  1エポックあたりのミニバッチ学習の回数
        rep_per_epoch = N // B
        # ミニバッチ学習の総回数
        rep = epoch * rep_per_epoch

        # 損失関数の和
        En_sum = 0

        from time import time
        t0 = time()

        for i in range(rep):
            # ミニバッチ学習
            mini_batch, En = self.mini_batch_training(X_train, Y_train, B)
            # ミニバッチの正解データ
            Y = mini_batch[1]

            # 逆伝播
            dEn_dX = self.layers['softmax'].backward(Y, B)
            dEn_dX = self.layers['affine2'].backward(dEn_dX)
            dEn_dX = self.layers['dropout'].backward(dEn_dX)
            dEn_dX = self.layers['relu2'].backward(dEn_dX)
            dEn_dX = self.layers['bn2'].backward(dEn_dX)
            dEn_dX = self.layers['affine1'].backward(dEn_dX)

            # Pooling と ReLU
            dEn_dX = self.layers['pooling'].convert_vectors_into_images(dEn_dX)
            dEn_dX = self.layers['pooling'].backward(dEn_dX)
            dEn_dX = self.layers['relu1'].backward(dEn_dX)

            # Batch Normalization
            B, ch, dh, dw = dEn_dX.shape
            # (dw, dh * ch, B)
            dEn_dX = dEn_dX.transpose(1, 2, 3, 0).reshape(-1, B)
            dEn_dX = self.layers['bn1'].backward(dEn_dX)
            # (B, ch, dh, dw)
            dEn_dX = dEn_dX.reshape(ch, dh, dw, B).transpose(3, 0, 1, 2)

            dEn_dX = self.layers['conv'].backward(dEn_dX)


            # 1エポック終わったら、Enの平均を出力
            if (i + 1) % rep_per_epoch == 0:

                self.epoch_count += 1
                En_avg = En_sum / rep_per_epoch



                En_sum = 0

                accuracy_train = self.check_accuracy(X_train, Y_train)
                accuracy_test = self.check_accuracy(X_test, Y_test)
                t1 = time()
                print(self.epoch_count, En_avg, accuracy_train, accuracy_test, (t1 - t0) / 60)
                t0 = time()

                ten = (i + 1) // rep_per_epoch
                if ten % 10 == 0:
                    self.save('mnist_' + str(ten) + '.npz')
            else:
                En_sum += En

    # 性能評価
    def check_accuracy(self, X, Y):
        N = len(X)
        B = 1000
        correct = sum(np.sum(self.forward(X[i:i+B]).argmax(axis = 0) == Y[i:i+B]) for i in range(0, N, B))
        # wrong = N - correct

        return correct / N


# 乱数配列
def random_ndarray(N, shape):
    return np.random.normal(0, 1 / np.sqrt(N), shape)

# ネットワーク層生成
def create_layers():
    return {'conv': Convolution(),
            'bn1': Batch_normalization(),
            'relu1': ReLU(),
            'pooling': Pooling(),
            'affine1': Affine(),
            'bn2': Batch_normalization(),
            'relu2': ReLU(),
            'dropout': Dropout(),
            'affine2': Affine(),
            'softmax': Softmax()}

# ミニバッチ生成
def create_mini_batch(X, Y, B):
    N = len(X)
    indexes = np.random.choice(N, B, False)
    X = X[indexes]
    Y = Y[indexes] # 1次元
    # Y = Y[indexes].T
    return X, Y

# クロスエントロピー誤差
def cross_entropy(y, y2):
    B = y2.shape[1]
    cols = np.arange(B)
    E = -np.log(y2[y, cols])
    return E
    # E = -y * np.log(y2)
    # return np.sum(E, axis = 0)

# (B, ch, dy, dx) に対してパディング
def padding(array, p):
    return np.pad(array, ((0, 0), (0, 0), (p, p), (p, p)))

# 画像バッチから行列Xへの変換
def convert_batch_into_X(batch, R, p, s):
    # バッチサイズ，チャンネル，縦，横
    B, ch, dy, dx = batch.shape

    # パディング
    batch = padding(batch, p)

    # フィルタがずれる距離
    di = 2 * p + dy - R
    dj = 2 * p + dx - R

    # 出力画像のサイズ dw × dh
    dh = di // s + 1
    dw = dj // s + 1

    X = np.empty((B, ch, R, R, dh, dw))
    for i in range(R):
        for j in range(R):
            X[:, :, i, j, :, :] = batch[:, :, i:i+di+1:s, j:j+dj+1:s]

    # (R * R * ch, dw * dh * B) の X と次元数 (B, dh, dw) を返す
    X = X.transpose(1, 2, 3, 0, 4, 5).reshape(R * R * ch, -1)

    return X, (B, dh, dw)

# 行列Xから画像バッチへの変換
def convert_X_into_batch(X, R, p, s, shape):
    B, dh, dw = shape

    di = s * (dh - 1)
    dj = s * (dw - 1)

    dy = di - 2 * p + R
    dx = dj - 2 * p + R


    # (B, ch, R, R, dh, dw)
    X = X.reshape(-1, R, R, B, dh, dw).transpose(3, 0, 1, 2, 4, 5)

    ch = X.shape[1]
    batch = np.zeros((B, ch, dy + 2 * p + s - 1, dx + 2 * p + s - 1))

    for i in range(R):
        for j in range(R):
            batch[:, :, i:i+di+1:s, j:j+dj+1:s] += X[:, :, i, j, :, :]

    # パディングを除く
    batch = batch[:, :, p:dy+p, p:dx+p]
    return batch
