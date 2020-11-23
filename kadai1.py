from mnist_tool import *
from neural_network import Neural_network

def main():
    # テストデータの読み込み
    X = load_file(test_images_file)

    # 入力
    i = int(input('0 ~ 9999から1つ入力してください >> '))

    # 前処理
    x = X[i].reshape((1, 1, 28, 28))

    # ニューラルネットワーク生成
    nn = Neural_network()
    nn.init(d, M, C, ch, K, R, p, s, d2)

    # 順伝播
    y = nn.forward(x)

    # 後処理
    y = post_process(y)
    print(y)

if __name__ == '__main__':
    main()
