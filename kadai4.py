from mnist_tool import *
from neural_network import Neural_network
import matplotlib.pyplot as plt
from pylab import cm

def main():
    # テストデータの読み込み
    X = load_file(test_images_file)
    Y = load_file(test_labels_file)

    # 入力
    i = int(input('0 ~ 9999から1つ入力してください >> '))

    # 前処理
    x = X[i].reshape((1, 1, 28, 28)) / 255

    # ニューラルネットワーク生成
    nn = Neural_network()
    nn.load(parameters_file)

    # 順伝播
    y = nn.forward(x)
    print(y)

    # 後処理
    y = post_process(y)

    # 結果の表示
    print('出力:',  y)
    plt.imshow(X[i], cmap = cm.gray)
    plt.show()

if __name__ == '__main__':
    main()
