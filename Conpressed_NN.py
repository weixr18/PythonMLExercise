import sys
import glob
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

PIC_NUM = 300
PIC_SCALE = 48


class CompressedNearestNeighbor():
    def __init__(self, k=3, metric="minkovskey", p=2):
        """
        压缩k近邻法
        def __init__(self, k=3, metric="minkovskey", p=2)
        """
        self.metric = metric
        self.k = k
        self.X_G = []
        self.X_S = []
        self.Y_G = []
        self.Y_S = []
        self.__N = 0
        self.__TRAIN_CONTINUE = False
        self.__p = p

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        self.__N = len(X)

        r = random.randint(0, self.__N - 1)
        self.X_G.append(list(X[r]))
        self.Y_G.append(Y[r])
        self.X_S.clear()
        self.Y_S.clear()
        self.X_S.extend([list(X[i]) for i in range(self.__N) if i != r])
        self.Y_S.extend([Y[i] for i in range(self.__N) if i != r])

        self.__TRAIN_CONTINUE = True
        round = 0

        while self.__TRAIN_CONTINUE:
            self.__TRAIN_CONTINUE = False
            for x, y in zip(self.X_S[:], self.Y_S[:]):
                y_pre = self.predict_one(np.array(x))
                if y != y_pre:
                    self.X_G.append(x)
                    self.Y_G.append(y)
                    i = self.X_S.index(list(x))
                    self.X_S.pop(i)
                    self.Y_S.pop(i)
                    self.__TRAIN_CONTINUE = True
            round += 1
            if (round % 1 == 0):
                print("Round: ", round)
        print("Conpressed count: ", len(self.X_G))

    def score(self, X, Y):
        Y_pre = [self.predict_one(x) for x in X]
        Y_pre = np.array(Y_pre)
        acc = np.sum(np.equal(Y_pre, Y))
        return acc/len(Y)

    def euclidean(self, x, y):
        return np.sqrt(np.sum(np.square(x - y)))

    def manhattan(self, x, y):
        return np.sum(np.abs(x - y))

    def minkovskey(self, x, y):
        if self.__p == 2:
            return self.euclidean(x, y)
        elif self.__p == 1:
            return self.manhattan(x, y)
        else:
            return np.power(np.sum(np.power(x - y, p)), 1 / p)

    def cosine(self, x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * (np.linalg.norm(y)))

    def chebyshev(self, x, y):
        return np.abs(x - y).max()

    def distance(self, x, y):
        """
        距离计算                    
        可用参数：                  
            "euclidean"             
            "manhatton"             
            "mincovskey"(default)   
            "cosine"                
            "chebyshev"             
        """
        d = self.__getattribute__(self.metric)
        return d(x, y)

    def predict_one(self, x):
        x = np.array(x)
        d_and_ys = [[self.distance(x, x_g), y_g]
                    for (x_g, y_g) in zip(self.X_G, self.Y_G)]
        d_and_ys = np.array(d_and_ys)
        d_and_ys = d_and_ys[d_and_ys[:, 0].argsort()]

        votes = d_and_ys[:self.k, 1]
        votes = votes.astype(np.int32)
        return np.argmax(np.bincount(votes))


def high_light_mess(mess):
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
    Datapaths = [("./data/Pictures/" + str(rands[i]) + "/*.png")
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
    # 增广
    o = np.ones(class_num * PIC_NUM)
    X = np.c_[X, o]

    return X, Y


if __name__ == "__main__":

    # 获取命令行参数
    CLASS_NUM = int(sys.argv[1])
    classes = None
    if (len(sys.argv) >= 3 and sys.argv[2] is not ""):
        classes = sys.argv[2].split(",")

    print("开始训练，请确保图片位于./data/Pictures路径中")
    if (CLASS_NUM > 1):
        print("分类器训练较慢，请耐心等待...")

    # 读取数据
    X, Y = read_data(CLASS_NUM, classes=classes)
    train_X, test_X, train_y, test_y = train_test_split(X, Y,
                                                        test_size=0.25,
                                                        random_state=0)

    # 开始训练
    param = {
        'k': 1,
        'p': 1,
        'metric': "euclidean",
    }
    cnn = CompressedNearestNeighbor(**param)
    cnn.fit(train_X, train_y)

    # 进行评估
    acc = cnn.score(test_X, test_y)
    print(param)
    high_light_mess(str("From Compressed_NN: ") + str("Accuracy: ") + str(acc))
