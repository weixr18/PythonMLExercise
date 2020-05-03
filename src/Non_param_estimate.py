import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def k(x1, x2, sigma=1):
    """高斯核函数"""
    return np.exp(-1 * ((x1 - x2) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)


def train_parzenW(X, Y):
    """parzen窗估计"""
    X_p = []
    X_n = []
    for x, y in zip(X, Y):
        if y == 1:
            X_p.append(x)
        else:
            X_n.append(x)

    # 只准备数据，惰性求值
    def p_pos(a):
        p = 0
        for xp in X_p:
            p += k(a, xp)
        p /= len(X_p)
        return p

    def p_neg(a):
        p = 0
        for xn in X_n:
            p += k(a, xn)
        p /= len(X_n)
        return p

    return p_pos, p_neg


def min_error_predict(X, Y, p_pos, p_neg):
    """最小错误率预测"""
    p_cor = 0
    n_cor = 0
    p_sum = 0
    n_sum = 0
    for x, y in zip(X, Y):
        """
        由于没有先验（或两类先验概率都是0.5）
        所以类条件密度大的后验概率就大
        根据最小错误率决策，直接选择后验概率最大的
        也就是类条件密度大的一类作为决策
        """
        pp = p_pos(x)
        pn = p_neg(y)
        if (pp > pn):
            p_sum += 1
            if (y == 1):
                p_cor += 1
        elif (pp <= pn):
            n_sum += 1
            if (y == 0):
                n_cor += 1

    print("-----------minimum error-----------")
    print("sensitivity:", p_cor/p_sum)
    print("singularity:", n_cor/n_sum)
    print("accuracy:", (p_cor + n_cor) / (p_sum + n_sum))


def min_risk_predict(X, Y, p_x_pos, p_x_neg):
    p_cor = 0
    n_cor = 0
    p_sum = 0
    n_sum = 0

    for x, y in zip(X, Y):
        pp = p_x_pos(x)
        pn = p_x_neg(y)
        p_pos_x = pp / (pp + pn)
        """
        由于没有先验（或两类先验概率都是0.5）
        所以后验概率就是该类的p(x|w)除以两个p(x|w)之和
        p_pos = pp / (pp + pn)
        p_neg = 1 - p_pos_x
        R_pos = 1 * p_neg
        R_neg = 10 * p_pos
        R_pos < R_neg判为正类，即1*(1 - p_pos) < 10 * p_pos，
        即p_pos > 1/11时判为正类
        """

        if y == 1:
            p_sum += 1
            if p_pos_x * 11 > 1:
                p_cor += 1
        elif y == 0:
            n_sum += 1
            if p_pos_x * 11 <= 1:
                n_cor += 1

    print("-----------minimum risk-----------")
    print("sensitivity:", p_cor/p_sum)
    print("singularity:", n_cor/n_sum)
    print("accuracy:", (p_cor + n_cor) / (p_sum + n_sum))


if __name__ == "__main__":
    # 准备数据
    X_p = np.random.normal(-2.5, 1, 250)
    X_n = np.random.normal(2.5, 2, 250)
    X = np.r_[X_p, X_n]
    Y = []
    Y.append([1 for i in range(250)])  # 正样本
    Y.append([0 for i in range(250)])  # 负样本
    Y = np.array(Y)
    Y = Y.reshape([500, ])

    train_X, test_X, train_y, test_y = train_test_split(X, Y,
                                                        test_size=0.25,
                                                        random_state=0)

    # 开始训练
    p_pos, p_neg = train_parzenW(train_X, train_y)

    xs = np.arange(-8, 10, 0.05)
    ys_p = [p_pos(x) for x in xs]
    ys_n = [p_neg(x) for x in xs]
    plt.plot(xs, ys_p, 'r', label="Positive")
    plt.plot(xs, ys_n, 'b', label="Negative")
    plt.title("Parzen window estinate")
    plt.legend()
    plt.grid()
    plt.show()

    # 最小错误率预测
    min_error_predict(test_X, test_y, p_pos, p_neg)

    # 最小风险预测
    min_risk_predict(test_X, test_y, p_pos, p_neg)
