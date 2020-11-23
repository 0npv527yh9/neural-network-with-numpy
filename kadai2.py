from mnist_tool import *
from neural_network import Neural_network

def main():
    # 教師用データの読み込み
    X = load_file(train_images_file)
    Y = load_file(train_labels_file)

    # 前処理
    X = pre_process(X)
    # X, Y = pre_process(X, Y)

    # ニューラルネットワーク生成
    nn = Neural_network()
    nn.init(d, M, C, ch, K, R, p, s, d2)
    # nn.load(parameters_file)

    # ミニバッチ学習
    B = 128
    En = nn.mini_batch_training(X, Y, B)[1]
    print(En)

if __name__ == '__main__':
    main()
