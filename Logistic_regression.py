import numpy as np

LR_V = 1e-1
ROUND = 4000
DECAY_STEP = 800
DECAY_RATE = 0.0005
DEBUG = False


def sigmoid(x):
    return 1/(1+np.exp(-x))


def test_model(X, Y, d=2, N=100, w=None, broaden=False):

    if broaden:
        o = np.ones(N) / 10
        new_X = np.c_[X, o]
        new_Y = 0.1 + 0.8 * Y
    else:
        new_X = X
        new_Y = Y

    if w is not None:
        correct = 0
        loss = 0
        index = range(N)
        for x, y, i in zip(new_X, new_Y, index):
            f = sigmoid(np.sum(w * x))
            y_pre = 0.9 if f > 0.5 else 0.1
            if y_pre == y:
                correct += 1
            loss += (f - y)*(f - y)
        return correct / N, loss
    else:
        print("The weighting matrix should not be empty.")
        return None


def logistic_reg(X, Y, d=2, N=100, broaden=False):
    """
    逻辑斯蒂回归[解析解]                        \n
    X:          样本集, [N*d]            \n
    Y:          标签集, [N]              \n
    d:          样本维度数量             \n
    N:          样本数量                 \n
    broaden:    是否对输入向量进行增广    \n

    """
    if N < d:
        print("Error: The number of samples must be greater than the dimension.")
        return None

    pass


def logistic_reg(X, Y, d=2, N=100, ROUND=ROUND, verbose=False, broaden=False):
    """
    逻辑斯蒂回归[梯度下降解]                        \n
    X:          样本集, [N*d]            \n
    Y:          标签集, [N]              \n
    d:          样本维度数量             \n
    N:          样本数量                 \n
    ROUND:      训练轮数                 \n
    verbose:    是否显示训练过程          \n
    broaden:    是否对输入向量进行增广    \n

    """

    # 向量增广
    if broaden:
        o = np.ones(N) / 10
        new_X = np.c_[X, o]
    new_Y = 0.1 + 0.8 * Y

    # 准备参数
    w = np.random.rand(d + 1) * 0.01

    # k-fold交叉验证
    k = 10
    X_f = []
    Y_f = []
    fold_index = range(k)
    for i in fold_index:
        X_part = new_X[i * N // k:((i + 1) * N + 1) // k]
        X_f.append(X_part)
        Y_part = new_Y[i * N // k:((i + 1) * N + 1) // k]
        Y_f.append(Y_part)

    # 训练日志
    log = []

    for j in range(ROUND):
        # 除去第j%k折，其余参与训练
        if (j > DECAY_STEP):
            LR = LR_V * np.exp(-DECAY_RATE * (j - DECAY_STEP))
        else:
            LR = LR_V

        acc = 0
        for X_p, Y_p, i in zip(X_f, Y_f, fold_index):
            if (i == j % k):
                pass
            else:
                for x, y in zip(X_p, Y_p):
                    # 梯度下降
                    f_w_dot_x = sigmoid(np.sum(w * x))
                    dL_dw = (f_w_dot_x - y) * f_w_dot_x * (1 - f_w_dot_x) * x
                    w -= dL_dw * LR

            accuracy, loss = test_model(
                X_f[i % k], Y_f[i % k], d, len(X_f[i % k]), w)
            acc += accuracy

        acc /= k
        if verbose:
            if j % 100 == 0:
                print("round: ", j, "accuracy:", format(
                    acc, '.8f'), "loss:", loss, "LR", format(LR, '.6f'))

            if j == ROUND - 10:
                print("w:", w)
    return w


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

    logistic_reg(X, Y, len(X[0]), len(X), broaden=True)
