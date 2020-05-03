import numpy as np
import matplotlib.pyplot as plt
import math
import sys


def MLEEstimate(X):
    mean = np.mean(X)
    var = np.var(X)
    return mean, var


def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)


def U_distribution(x, low, high):
    if (high < low):
        t = high
        high = low
        low = t

    return [(1 / (high - low) if xx >= low and xx <= high else 0) for xx in x]


def MLE_normal_estimate_test():
    rounds = [10, 100, 1000]

    for r in rounds:
        es = []
        for _ in range(3):
            X = np.random.standard_normal(r)
            mean, var = MLEEstimate(X)
            es.append((mean, np.sqrt(var)))

        xs = np.arange(-5, 5, 0.01)
        ys = [normal_distribution(xs, *es[i]) for i in range(3)]
        colors = ['r', 'g', 'b']
        for i in range(3):
            plt.plot(xs, ys[i], colors[i], label="mean, sigma:" + str(es[i]))

        y_stand = normal_distribution(xs, 0, 1)

        plt.plot(xs, y_stand, 'black', label="N(0, 1)")
        plt.title("sample number:" + str(r))
        plt.legend()
        plt.grid()
        plt.show()


def Bayes_normal_estimate_test(range_2):
    N = 1000
    X = np.random.standard_normal(N)
    var = 1
    var_0s = [0.01 * var, 0.1 * var, var, 10 * var]
    mean_0 = -5

    m_N = np.mean(X)
    xs = np.arange(-range_2, range_2, range_2*2/1000)
    ys = []
    es = []
    for var_0 in var_0s:
        mean_e = N * var_0 / (N * var_0 + var) * m_N + \
            var / (N * var_0 + var) * mean_0
        ys.append(normal_distribution(xs, mean_e, np.sqrt(var)))
        es.append(mean_e)

    colors = ['r', 'g', 'b', 'gray']
    for i in range(4):
        str1 = "mean: " + str(round(es[i], 4)) + \
            " var_0: " + str(round(var_0s[i], 4))
        plt.plot(xs, ys[i], colors[i], label=str1)

    y_stand = normal_distribution(xs, 0, 1)

    plt.plot(xs, y_stand, 'black', label="N(0, 1)")
    plt.legend()
    plt.grid()
    plt.show()


def U_normal_estimate_test():
    rounds = [10, 100, 1000]

    for r in rounds:
        es = []
        for _ in range(3):
            X = np.random.uniform(low=0, high=1, size=r)
            mean, var = MLEEstimate(X)
            es.append((mean, np.sqrt(var)))

        xs = np.arange(-5, 5, 0.01)
        ys = [normal_distribution(xs, *es[i]) for i in range(3)]
        colors = ['r', 'g', 'b']
        for i in range(3):
            plt.plot(xs, ys[i], colors[i], label="mean, sigma:" + str(es[i]))

        y_stand = U_distribution(xs, 0, 1)

        plt.plot(xs, y_stand, 'black', label="N(0, 1)")
        plt.title("sample number:" + str(r))
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    range_2 = 5
    if len(sys.argv) > 1:
        range_2 = float(sys.argv[1]) if (float(sys.argv[1]) > 0) else 5

    # (1)
    MLE_normal_estimate_test()

    # (2)
    Bayes_normal_estimate_test(range_2)

    # (3)
    U_normal_estimate_test()
