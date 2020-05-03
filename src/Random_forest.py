import sys
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

PIC_NUM = 300
PIC_SCALE = 48
DEBUG = False
DRAW_CURVE = True


def show2class(clf):
    """二类分类展示"""
    ppps = clf.predict_proba(test_X)
    # 将预测正确率和标签一起按照ppp排序
    samples = [[ppp[1], test_y[i]] for (i, ppp) in enumerate(ppps)]
    samples = np.array(samples)
    samples = samples[samples[:, 0].argsort()]

    # 二分查找：arr中第一个大于x的元素的索引
    def binarySearch(arr, left, right, x):
        if (x < arr[0]):
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
    rec = [[1, 1, 1]]
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

    if DRAW_CURVE:
        plt.plot(FPRs, TPRs)
        plt.ylim(0, 1.05)
        plt.show()

    AUC = np.sum(TPRs) / len(TPRs)
    print("AUC:", AUC)

    # 计算阈值0.5时的TPR FPR等
    I = binarySearch(samples[:, 0], 0, len(samples), 0.5)
    s_p = samples[I:]
    s_n = samples[:I]

    TP = sum(s_p[:, 1] == 1)
    FP = sum(s_p[:, 1] == 0)
    TPR = TP / P_NUM
    FPR = FP / N_NUM
    FDR = TP / (TP + FP)

    mess = "Test accuracy:" + str((TP + N_NUM - FP) / (P_NUM + N_NUM))
    high_light_mess("From Random_forest: " + mess)

    print("TPR:", TPR)
    print("TNR:", 1 - FPR)
    print("FPR:", FPR)
    print("FNR:", 1 - TPR)
    print("FDR:", FDR)


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


def read_data(class_num, classes=None):
    """
    读取数据
    """
    # 随机抽取class_num组样本，或接受从命令行指定的样本组
    if classes is not None:
        rands = classes
    else:
        rands = random.sample(range(0, 10), class_num)
    print("Chosen faces:", rands)

    # 样本路径
    Datapaths = [("../data/Pictures/" + str(rands[i]) + "/*.png")
                 for i in range(class_num)]
    X = np.zeros((class_num * PIC_NUM, PIC_SCALE, PIC_SCALE))
    Y = np.zeros(class_num * PIC_NUM)

    # 随机序号，为了打乱读入的样本
    rand_list = random.sample(
        range(0, class_num * PIC_NUM), class_num * PIC_NUM)
    rand_list = np.array(rand_list).reshape([class_num, PIC_NUM])

    # 读入样本和标签，并把标签赋值为新的one-hot
    for j, (Datapath, rands) in enumerate(zip(Datapaths, rand_list)):
        # j : 新的
        for index, imageFile in enumerate(glob.glob(Datapath)):
            img = np.array(Image.open(imageFile).convert('L'))
            X[rands[index]] = img
            Y[rands[index]] = j

    # 扁平化和归一化
    X = X.reshape((class_num * PIC_NUM, PIC_SCALE * PIC_SCALE))
    X = X / 256

    o = np.ones(class_num * PIC_NUM)
    X = np.c_[X, o]

    return X, Y


if __name__ == "__main__":

    # 读取数据
    CLASS_NUM = int(sys.argv[1])

    classes = None
    if (len(sys.argv) >= 3 and sys.argv[2] is not ""):
        classes = sys.argv[2].split(",")

    if (len(sys.argv) >= 4):
        if sys.argv[3] == "D":
            DEBUG = True
        if sys.argv[3] == "F":
            DRAW_CURVE = False

    if (CLASS_NUM > 2):
        print("多类分类器训练较慢，请耐心等待...")

    X, Y = read_data(CLASS_NUM, classes=classes)

    if DEBUG:
        """调试模式"""
        # 准备参数
        """
        paramset = {
            'n_estimators': [10, 50, 100, 500, 1000],
            'max_depth': [10, 50, 100, 500, 1000],
            'max_features': ['sqrt'],
        }
        """
        paramset = {
            'n_estimators': [500],
            'max_depth': [10, 20, 50, 100, 200, 500],
            # 'max_features': ['sqrt'],
            # 'min_samples_split': [2],
            # 'max_leaf_nodes': [100000],
        }

        params = []

        for key in paramset.keys():
            pt = {}
            n_params = []
            for v in paramset[key]:
                pt.update({key: v})
                if len(params) > 0:
                    for p in params:
                        pt_c = pt.copy()
                        pt_c.update(p)
                        n_params.append(pt_c)
                else:
                    n_params.append(pt.copy())
            params = n_params

        # 逐个训练并展示结果
        for param in params:
            m = 0
            for i in range(5):
                train_X, test_X, train_y, test_y = train_test_split(X,
                                                                    Y,
                                                                    test_size=0.25)
                rfc = RandomForestClassifier(**param)
                rfc.fit(train_X, train_y)
                res = rfc.score(test_X, test_y)
                print(res)
                m += res
            m /= 5

            print(param)
            mess = "From Random_forest: " + " Accuracy: " + str(m)
            high_light_mess(mess)

    else:
        """训练模式"""
        train_X, test_X, train_y, test_y = train_test_split(X,
                                                            Y,
                                                            test_size=0.25)

        param = {
            'n_estimators': 500,
            'max_depth': 500,
            'min_samples_split': 2,
            'max_leaf_nodes': 100000,
        }

        if CLASS_NUM == 2:
            rfc = RandomForestClassifier(**param)
            rfc.fit(train_X, train_y)
            show2class(rfc)
        else:
            rfc = RandomForestClassifier(**param)
            rfc.fit(train_X, train_y)
            m = rfc.score(test_X, test_y)
            print(param)
            mess = "From Random_forest: " + " Accuracy: " + str(m)
            high_light_mess(mess)
