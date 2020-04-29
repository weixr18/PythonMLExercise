import numpy as np
from sklearn.linear_model import Lasso


LR = 1e-4
TRAIN_ROUND = 100
DEBUG = True


def linear_reg(X, Y, d, N, X_t=[], Y_t=[]):
    """
    线性回归[解析解]              \n
    X: 样本集, [N*d]             \n
    Y: 标签集, [N]              \n
    d: 样本维度数量              \n
    N: 样本数量                 \n

    """
    w = np.dot(X.T, X)
    w = np.dot(np.linalg.inv(w), X.T)
    w = np.dot(w, Y)

    return w


def ridge_reg(X, Y, d, N, X_t=[], Y_t=[], _lambda=1):
    """
    岭回归[解析解]              \n
    X: 样本集, [N*d]             \n
    Y: 标签集, [N]              \n
    d: 样本维度数量              \n
    N: 样本数量                 \n

    """
    w = np.dot(X.T, X)
    w += _lambda * np.identity(X.shape[1])
    w = np.dot(np.linalg.inv(w), X.T)
    w = np.dot(w, Y)
    return w


def lasso_reg(X, Y, d, N, X_t=[], Y_t=[], _lambda=1):
    """
    岭回归[梯度下降解]              \n
    X: 样本集, [N*d]             \n
    Y: 标签集, [N]              \n
    d: 样本维度数量              \n
    N: 样本数量                 \n

    """
    model = Lasso(alpha=_lambda)
    model.fit(X, Y)
    return model.coef_


def get_data():
    x1 = np.arange(1, 21, 1, dtype='float64')
    y = 3 * x1 + 2
    y += np.random.normal(0, np.sqrt(2), 20)
    x2 = x1 * 0.05
    x2 += np.random.normal(0, np.sqrt(0.5), 20)
    X = np.array([x1, x2]).T

    print("corr coefficent:", np.corrcoef(X.T)[0][1])

    return X, y


if __name__ == "__main__":

    ROUND = 10
    w = []
    for _ in range(ROUND):
        # train
        X, Y = get_data()

        w1 = linear_reg(X, Y, 4, len(X))
        w2 = ridge_reg(X, Y, 4, len(X))
        w3 = lasso_reg(X, Y, 4, len(X))

        ws = [w1, w2, w3]
        print(ws)
        w.append(ws)

    w = np.array(w)
    w = np.swapaxes(w, 0, 1)
    var = np.var(w, axis=1)
    print(var)
    mean = np.mean(w, axis=1)
    print(mean)

    ROUND = 10
    w = []
    X, Y = get_data()

    for l in range(1, ROUND + 1):
        # train
        w2 = ridge_reg(X, Y, 4, len(X), _lambda=l)
        print(list(w2), np.sum(np.square(w2)))

    for l in range(1, ROUND + 1):
        w3 = lasso_reg(X, Y, 4, len(X), _lambda=l)
        print(list(w3), np.sum(np.square(w3)))

    pass
