from cifar_tool import *
from neural_network import Neural_network
import matplotlib.pyplot as plt

def main():
    # 教師用データの読み込み
    X, Y = load_train_file()

    # 入力
    i = int(input('0 ~ 9999から1つ入力してください >> '))

    # 前処理
    x = X[i][np.newaxis, :, :, :] / 255

    # ニューラルネットワーク生成
    nn = Neural_network()
    nn.load(parameters_file)

    # 順伝播
    y = nn.forward(x)

    # 後処理
    y = post_process(y)

    # 結果の表示
    print('出力:',  label[y])
    print('正解:',  label[Y[i]])
    plt.imshow(X[i].transpose(((1,2,0))))
    plt.show()

if __name__ == '__main__':
    main()
