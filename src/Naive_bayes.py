import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

epsilon = 1e-8
warnings.filterwarnings("ignore", category=Warning)


def softmax(a):
    """softmax函数"""
    exp_a = np.exp(a)
    return exp_a/np.sum(exp_a)


class NaiveBayes():
    """
        朴素贝叶斯分类器                         \n
        采用高斯（正态）先验                     \n
        参数估计使用MLE                         \n
        假设特征每一维度彼此独立，                \n
        每特征每类分别服从正态分布                \n
        d   :   特征维数                        \n
        c   :   类别数                          \n
    """

    def __init__(self, d, c=2):
        self.d = d
        self.c = c
        self.μ = np.zeros([d, c])  # 类条件密度均值
        self.σ = np.zeros([d, c])  # 类条件密度标准差
        self.p_prior = np.zeros(c)  # 先验分布

    def fit(self, X, Y):
        """训练"""
        for j in range(self.c):
            self.p_prior[j] = len(Y[np.where(Y == j)]) / len(Y)
            for i in range(self.d):
                X_i = X[:, i]
                X_i_Cj = X_i[np.where(Y == j)]
                self.μ[i, j] = np.mean(X_i_Cj)
                self.σ[i, j] = np.var(X_i_Cj) + epsilon

    def predict_one(self, x):
        """预测"""
        p_x_c = np.zeros([self.d, self.c])

        for i in range(self.d):
            for j in range(self.c):
                p_x_c[i, j] = self.normal_distribution(
                    x[i], self.μ[i, j], self.σ[i, j])
        """
        类条件概率即为每个特征该类条件概率之积。
        用对数求和代替求积防止概率过小
        """
        lg_p_x_c = np.log(p_x_c + epsilon)  # 防止预测值出现0无法取log
        """
        后验概率为先验概率乘以类条件密度，再归一化
        等价于对数相加再softmax
        """
        lg_p_c_x = np.sum(lg_p_x_c, axis=0)
        a = np.log(self.p_prior + epsilon)
        lg_p_c_x += np.log(self.p_prior + epsilon)

        # 归一化
        p_c_x = softmax(lg_p_c_x)
        return p_c_x

    def predict_proba(self, X):
        res = [self.predict_one(x) for x in X]
        return np.array(res)

    def score(self, X, Y):
        """测试"""
        Y_pre = [self.predict_one(x) for x in X]
        Y_pre = np.argmax(np.array(Y_pre), axis=1)
        acc = np.sum(np.equal(Y_pre, Y)) / len(Y)
        return acc

    def normal_distribution(self, x, mean, sigma):
        return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (np.sqrt(2 * np.pi) * sigma)


def show_2class_res(test_X, test_y, model):
    """二分类结果展示"""
    ppps = model.predict_proba(test_X)

    # 将预测正确率和标签一起按照ppp排序
    samples = [[ppp[1], test_y[i]]
               for (i, ppp) in enumerate(ppps)]
    samples = np.array(samples)
    samples = samples[samples[:, 0].argsort()]

    # 二分查找：arr中第一个大于x的元素的索引
    def binarySearch(arr, left, right, x):
        if(x < arr[0]):
            return 0
        while left + 1 < right:
            mid = int(left + (right - left) / 2)
            if x >= arr[mid]:
                left = mid
            else:
                right = mid
        return left + 1

    P_NUM = sum(samples[:, 1] == 1)
    N_NUM = len(samples) - P_NUM

    # 计算各阈值下的TPR和FPR
    t = 0
    rec = [(1, 1, 1)]
    while t <= 1:
        I = binarySearch(samples[:, 0], 0, len(samples), t)
        s_p = samples[I:]
        s_n = samples[:I]

        TP = sum(s_p[:, 1] == 1)
        FP = sum(s_p[:, 1] == 0)

        TPR = TP / P_NUM
        FPR = FP / N_NUM

        rec.append([t, TPR, FPR])
        t += 0.02
    rec.append([0, 0, 0])

    # 画ROC曲线
    rec = np.array(rec)
    FPRs = rec[:, 2].reshape(len(rec))
    TPRs = rec[:, 1].reshape(len(rec))
    DRAW_CURVE = True
    if DRAW_CURVE:
        plt.plot(FPRs, TPRs)
        plt.ylim(0, 1.05)
        plt.show()

    AUC = np.sum(TPRs) / len(TPRs)

    # 计算阈值0.5时的TPR FPR等
    I = binarySearch(samples[:, 0], 0, len(samples), 0.5)
    s_p = samples[I:]
    s_n = samples[:I]

    TP = sum(s_p[:, 1] == 1)
    FP = sum(s_p[:, 1] == 0)

    TPR = TP / P_NUM
    FPR = FP / N_NUM

    mess = "Test accuracy:" + str((TP + N_NUM - FP) / (P_NUM + N_NUM))
    high_light_mess("From Naive Bayes: " + mess)

    print("AUC:", AUC)
    print("TP:", TP, "FN:", P_NUM - TP)
    print("FP:", FP, "TN:", N_NUM - FP)

    print("TPR:", TPR, "FNR:", 1 - TPR)
    print("FPR:", FPR, "TNR:", 1 - FPR)


def high_light_mess(mess):
    """高亮显示"""
    import ctypes

    STD_OUTPUT_HANDLE = -11
    FOREGROUND_YELLOW = 0x0e  # yellow.
    FOREGROUND_BLUE = 0x09  # blue.
    FOREGROUND_GREEN = 0x0a  # green.
    FOREGROUND_RED = 0x0c  # red.

    std_out_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)

    def set_cmd_text_color(color, handle=std_out_handle):
        Bool = ctypes.windll.kernel32.SetConsoleTextAttribute(handle, color)
        return Bool

    def resetColor():
        set_cmd_text_color(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE)

    set_cmd_text_color(FOREGROUND_YELLOW)
    print(mess)
    resetColor()


def read_data():
    """
    读取数据
    """
    # 样本路径
    datapath = "../data/spambase/spambase.data"
    X = np.loadtxt(datapath, delimiter=',', dtype=float)

    Y = X[:, -1:]
    Y = Y.reshape(len(Y)).astype('int64')
    X = X[:, :-1]
    return X, Y


if __name__ == "__main__":
    X, Y = read_data()
    train_X, test_X, train_y, test_y = train_test_split(X,
                                                        Y,
                                                        test_size=1000)

    print("------My Gaussian-----")
    nb = NaiveBayes(X.shape[1], len(np.unique(Y)))  # 手写高斯先验
    nb.fit(train_X, train_y)
    show_2class_res(test_X, test_y, nb)

    print("------Multional-----")
    mnb = MultinomialNB()       # 多项式先验
    mnb.fit(train_X, train_y)
    show_2class_res(test_X, test_y, mnb)

    print("------HMMlearn Gaussian-----")
    gnb = GaussianNB()       # 高斯先验
    gnb.fit(train_X, train_y)
    show_2class_res(test_X, test_y, gnb)
