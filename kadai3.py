from mnist_tool import *
from neural_network import Neural_network

def main():
    # 教師用データの読み込み
    X_train = load_file(train_images_file)
    Y_train = load_file(train_labels_file)

    # テスト用データの読み込み
    X_test = load_file(test_images_file)
    Y_test = load_file(test_labels_file)

    # 前処理
    X_train = pre_process(X_train)
    X_test = pre_process(X_test)
    # X, Y = pre_process(X, Y)

    # ニューラルネットワーク生成
    nn = Neural_network()

    if input('再学習しますか >> [y,n]') == 'y':
        nn.load(parameters_file)
    else:
        nn.init(d, M, C, ch, K, R, p, s, d2)

    # バッチサイズ
    B = 128
    # エポック数
    epoch = int(input('エポック数 >> '))

    # 誤差伝播
    nn.backward(X_train, Y_train, X_test, Y_test, B, epoch)

    # パラメータを保存
    nn.save(parameters_file)

if __name__ == '__main__':
    main()
