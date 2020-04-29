import numpy as np
import math


def fisher(X, Y, d, N, print_res=True):
    """
    Fisher线性判别：二类分类     \n
    X: 样本集, [N*d]             \n
    Y: 标签集, [N]              \n
    d: 样本维度数量              \n
    N: 样本数量                 \n

    """
    # 类均值向量
    X_0 = [X[i] for i in range(N) if Y[i] == 0]
    X_1 = [X[i] for i in range(N) if Y[i] == 1]

    X_0 = np.array(X_0)
    X_1 = np.array(X_1)

    m_0 = np.mean(X_0, axis=0)
    m_1 = np.mean(X_1, axis=0)

    # 总类内离散度矩阵
    S_0 = [np.array([x - m_0]).T * (x - m_0) for x in X_0]
    S_0 = np.sum(S_0, axis=0)

    S_1 = [np.array([x - m_1]).T * (x - m_1) for x in X_1]
    S_1 = np.sum(S_1, axis=0)

    S_w = S_0 + S_1

    # 类间离散度矩阵
    S_b = np.array([m_0]).T * m_1

    # 最优投影方向
    S_w_I = np.matrix(S_w).I
    S_w_I /= np.mean(S_w_I)
    _w = S_w_I * np.array([m_0 - m_1]).T
    _w = np.array(_w).T
    _w = _w[0]

    # 最优分类阈值，假设各类分布服从高斯分布
    _w0 = (m_0 + m_1) * S_w_I * np.array([m_0 - m_1]).T * -0.5
    _w0 -= math.log(X_0.shape[0] / (N - X_0.shape[0]))

    # 最终判别函数
    def g(x): return np.dot(_w, x) + _w0

    # 显示结果
    if (print_res):
        G = [g(x) for x in X]
        G = [1 if g < 0 else 0 for g in G]
        acc = 0
        for y, gg in zip(Y, G):
            if (y == gg):
                acc += 1
        acc /= len(X)
        print("acc:", acc)
        print("w:", _w)
        print("w0:", _w0[0])
    return g


def valid_model(X, Y, d, N, g):
    correct = 0
    loss = 0
    index = range(N)
    for x, y, i in zip(X, Y, index):
        f = sigmoid(np.sum(w * x))
        y_pre = 0.9 if f > 0.5 else 0.1
        if y_pre == y:
            correct += 1
        loss += (f - y)*(f - y)
    return correct/N, loss


def getfloat(n):
    if n == '?':
        return - 1
    else:
        return float(n)


def read_data(path):
    data = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            data.append(line)
        data = [[getfloat(j) for j in i] for i in data]
        return data
    return None


if __name__ == "__main__":
    # 读入数据
    train = read_data("data/breast-cancer-wisconsin.txt")
    X = [d[1:10] for d in train]
    Y = [d[10] for d in train]
    X = np.array(X)
    Y = np.array(Y)

    # 缺失值用均值代替
    X_6 = X[:, 5:6]
    X_6_v = [list(x) for x in X_6 if x > 0]
    mean_6 = np.mean(X_6_v)
    X_6 = np.array([([mean_6] if x < 0 else x) for x in X_6])
    X_6 = X_6.reshape([len(X)])
    X[:, 5] = X_6

    g = fisher(X, Y, len(X[0]), len(X))
