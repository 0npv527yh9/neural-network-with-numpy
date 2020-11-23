from mnist_tool import *
from neural_network import Neural_network

def main():
    nn = Neural_network()
    nn.load(parameters_file)

    # 教師用データ
    X = load_file(train_images_file)
    Y = load_file(train_labels_file)
    X = pre_process(X)
    accuracy = nn.check_accuracy(X, Y)
    print('　訓練データ', accuracy)

    # テスト用データ
    X = load_file(test_images_file)
    Y = load_file(test_labels_file)
    X = pre_process(X)
    accuracy = nn.check_accuracy(X, Y)
    print('テストデータ', accuracy)

if __name__ == '__main__':
    main()
