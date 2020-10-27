import numpy as np
import _pickle as cp
import matplotlib.pyplot as plt


def main():
    X, y = cp.load(open('winequality-white.pickle', 'rb'))
    N, D = X.shape
    N_train = int(0.8 * N)
    N_test = N - N_train

    X_train = X[:N_train]
    y_train = y[:N_train]
    X_test = X[:N_test]
    y_test = y[:N_test]

    #draw_bar(y_train)
    print(compute_baseline(y_train, y_test))


def compute_baseline(y_train, y_test):
    # Calculates the mean of all the y-values in the training data and the MSE from that regarding the test data
    # Used as a baseline for other prediction models
    predictor = np.mean(y_train)
    mse = np.sum((y_test - predictor) ** 2) / len(y_test)
    return mse


def draw_bar(ys):
    y_groups = []
    for quality_level in range(3, 10):
        y_groups.append(len([1 for y in ys if y==quality_level]))
    x_pos = [i for i, _ in enumerate(y_groups)]

    plt.bar(x_pos, y_groups)
    plt.xticks(x_pos, range(3, 10))
    plt.xlabel('Quality level')
    plt.ylabel('Number of samples')
    plt.title('Distribution of wine quality levels')
    plt.show()


if __name__ == '__main__':
    main()