import sys
import numpy as np
import random
import glob
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split


PIC_NUM = 300
PIC_SCALE = 48

ROUND = 200
DECAY_STEP = 200
DECAY_RATE = 0.005
LR_V = 0.0001


def softmax(a):
    """softmax函数"""
    exp_a = np.exp(a)
    return exp_a/np.sum(exp_a)


def validate_model(X, Y, d, N, W, c):
    """评估模型"""
    correct = 0
    loss = 0
    index = range(N)
    for x, y, i in zip(X, Y, index):
        f = softmax(np.dot(W.T, x))
        y_pre = np.argmax(f)
        y_l = np.argmax(y)

        if y_pre == y_l:
            correct += 1
        loss += -np.dot(y, np.log(f))
    return correct / N, loss


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


def show_result(X, Y, W, c, DRAW_CURVE):
    """
    展示训练结果
    """
    if c == 2:
        # 二分类

        # 获取所有样本的预测正类概率ppp (predict positive probability)
        ppps = [softmax(np.dot(W.T, x))[1] for x in X]

        # 将预测正确率和标签一起按照ppp排序
        samples = [[ppp, np.argmax(Y[i])] for (i, ppp) in enumerate(ppps)]
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
        rec = []
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

        # 画ROC曲线
        rec = np.array(rec)
        FPRs = rec[:, 2].reshape(len(rec))
        TPRs = rec[:, 1].reshape(len(rec))
        if DRAW_CURVE:
            plt.plot(FPRs, TPRs)
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

        mess = "Test accuracy:" + str((TP + N_NUM - FP) / (P_NUM + N_NUM))
        high_light_mess("From Softmax: " + mess)

        print("TPR:", TPR)
        print("TNR:", 1 - FPR)
        print("FPR:", FPR)
        print("FNR:", 1 - TPR)

        # 梯度可视化
        """
        w0 = W.T[0][0:6912]
        w0 = w0.reshape((PIC_SCALE, PIC_SCALE))
        w0 -= np.min(w0)
        w0 = Image.fromarray(np.uint8(w0 / np.max(w0) * 256))
        w0.show()
        """

    else:
        # 多分类
        correct = 0
        index = range(len(X))
        for x, y, i in zip(X, Y, index):
            f = softmax(np.dot(W.T, x))
            y_pre = np.argmax(f)
            y_l = np.argmax(y)

            if y_pre == y_l:
                correct += 1
        mess = "Test accuracy:" + str(correct / len(X))
        high_light_mess("From Softmax: " + mess)
    pass


def softmax_reg(X, Y, N, d, c):
    """
    softmax回归                 \n
    X: 样本集, [N*d]             \n
    Y: 标签集, [N]              \n
    d: 样本维度数量              \n
    N: 样本数量                 \n
    c: 类别数量                 \n

    """

    W = np.random.rand(d, c) / 100

    # k-fold交叉验证
    k = 10
    X_f = []
    Y_f = []
    fold_index = range(k)
    fold_sz = (N // k + 1) if N % k != 0 else N // k
    for i in fold_index:
        X_part = X[i * fold_sz:(i + 1) * fold_sz]
        X_f.append(X_part)
        Y_part = Y[i * fold_sz:(i + 1) * fold_sz]
        Y_f.append(Y_part)

    for j in range(ROUND):
        # 学习率衰减
        if (j > DECAY_STEP):
            LR = LR_V * np.exp(-DECAY_RATE * (j - DECAY_STEP))
        else:
            LR = LR_V

        acc = 0
        for X_p, Y_p, i in zip(X_f, Y_f, fold_index):
            # 除去第j%k折，其余参与训练
            if (i == j % k):
                pass
            else:
                for x, y in zip(X_p, Y_p):
                    # 梯度下降
                    w_dot_x = np.dot(W.T, x)
                    dL_dw = (softmax(w_dot_x) - y).reshape((c, 1))
                    dL_dw = (dL_dw * np.array([x])).T
                    W -= dL_dw * LR

            accuracy, loss = validate_model(
                X_f[i % k], Y_f[i % k], d, len(X_f[i % k]), W, c)
            acc += accuracy

        acc /= k
        if c == 10 and j % 10 == 0:
            print("round: ", j, "accuracy:", format(
                acc, '.5f'), "loss:", format(loss, '.5f'), "LR", format(LR, '.6f'))

    return W


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
    Y = np.zeros((class_num * PIC_NUM, class_num))

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
            Y[rands[index]] = [(1 if i == j else 0) for i in range(class_num)]

    # 扁平化和归一化
    X = X.reshape((class_num * PIC_NUM, PIC_SCALE * PIC_SCALE))
    X = X / 256

    o = np.ones(class_num * PIC_NUM)
    X = np.c_[X, o]

    """
    此处要把Y转化为one-hot！！！
    """

    return X, Y


if __name__ == "__main__":

    CLASS_NUM = int(sys.argv[1])

    classes = None
    if (len(sys.argv) >= 3 and sys.argv[2] is not ""):
        classes = sys.argv[2].split(",")

    DRAW_CURVE = True
    if (len(sys.argv) >= 4 and sys.argv[3] is "F"):
        DRAW_CURVE = False

    X, Y = read_data(CLASS_NUM, classes)

    if (CLASS_NUM > 3):
        print("多类分类器训练较慢，请耐心等待...")
        ROUND = 200

    train_X, test_X, train_y, test_y = train_test_split(X, Y,
                                                        test_size=0.25,
                                                        random_state=0)

    W = softmax_reg(train_X, train_y, len(train_X), len(X[0]), CLASS_NUM)

    show_result(test_X, test_y, W, CLASS_NUM, DRAW_CURVE)
