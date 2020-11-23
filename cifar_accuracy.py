from cifar_tool import *
from neural_network import Neural_network

def main():
    nn = Neural_network()
    nn.load(parameters_file)

    # 教師用データ
    X, Y = load_train_file()
    X = pre_process(X)
    accuracy = nn.check_accuracy(X, Y)
    print('　訓練データ', accuracy)

    # テスト用データの読み込み
    X, Y = load_test_file()
    X = pre_process(X)
    accuracy = nn.check_accuracy(X, Y)
    print('テストデータ', accuracy)

if __name__ == '__main__':
    main()
