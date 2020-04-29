import numpy as np

LR = 1e-4
ROUND = 10000
DEBUG = True


def linear_reg(X, Y, d, N, X_t=[], Y_t=[]):
    """
    线性回归[梯度下降解]           \n
    X: 样本集, [N*d]             \n
    Y: 标签集, [N]              \n
    d: 样本维度数量              \n
    N: 样本数量                 \n

    """
    for x in X:
        x[d:] = [1]
    X = np.array(X)
    Y = np.array(Y)
    X_mean = np.sum(X, axis=0) / N

    w = np.random.rand(d + 1) / 10

    log = []

    for j in range(ROUND):
        # gradient decent (sample by sample)
        for x, y in zip(X, Y):
            dL_dw = (np.sum(w * x) - y) * x
            w -= dL_dw * LR

        if j % 10 == 0:
            r_ind = np.random.randint(0, N, size=[N//10])
            X_val = np.array([X[i] for i in r_ind])
            Y_val = np.array([Y[i] for i in r_ind])
            Loss = np.dot(X_val, np.array([w]).T)
            Loss = Loss.reshape(N // 10) - Y_val
            Loss = np.sum(Loss * Loss)
            log.append(Loss)

        if DEBUG:
            if j % 100 == 0:
                test_model(X_t, Y_t, 4, w, j)
                pass

    # print(log)
    return w


def test_model(X_t, Y_t, d, w, ro="End"):
    for x in X_t:
        x[d:] = [1]
    X = np.array(X_t)
    Y = np.array(Y_t)

    SE = 0
    cmp = []
    for x, y in zip(X, Y):
        f = np.sum(w * x)
        SE += (f - y) * (f - y)
        cmp.append([f, y])
    if DEBUG:
        print("round : ", ro, "Test SE: ", round(SE, 5), "W = ", list(w))
    else:
        print("Test SE: ", round(SE, 5))  # , "W = ", list(w))

    return cmp


def read_data(path):
    data = []
    with open(path, "r") as f:
        heads = f.readline()
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            data.append(line)
        data = [[float(j) for j in i] for i in data]
        heads = heads.split('\t')
        return heads, data
    pass


if __name__ == "__main__":
    # prepare data
    heads, train = read_data(".\\data\\prostate_train.txt")
    X = [d[:4] for d in train]
    Y = [d[4] for d in train]

    heads, test = read_data(".\\data\\prostate_test.txt")
    X_t = [d[:4] for d in test]
    Y_t = [d[4] for d in test]

    # train
    w = linear_reg(X, Y, 4, len(X), X_t, Y_t)

    # test
    cmp = test_model(X_t, Y_t, 4, w)

    """
    for c in cmp:
        print('|', c[0], '|', c[1], '|')
    """
