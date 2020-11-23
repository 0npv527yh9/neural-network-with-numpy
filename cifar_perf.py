from cifar_tool import *
from neural_network import Neural_network

def output_performance(title, X, Y, nn):
    X, _ = pre_process(X, Y)

    N = len(X)
    B = 1000
    correct = sum(np.sum(nn.forward(X[i:i+B]).argmax(axis = 0) == Y[i:i+B]) for i in range(0, N, B))
    # correct = sum(nn.forward(X.T).argmax(axis = 0) == Y)
    wrong = N - correct

    print(title)
    print('正: {:>5d}個 {:>5.2f}%'.format(correct, correct / N * 100))
    print('誤: {:>5d}個 {:>5.2f}%'.format(wrong, wrong / N * 100))

def main():
    nn = Neural_network()
    nn.load(parameters_file)

    # 教師用データ
    X, Y = load_train_file()
    output_performance('訓練データ', X, Y, nn)

    # テスト用データの読み込み
    X, Y = load_test_file()
    output_performance('テストデータ', X, Y, nn)

if __name__ == '__main__':
    main()
