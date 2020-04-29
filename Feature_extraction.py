import numpy as np

from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import argparse


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


def get_data():
    data = np.load('./data/mnist.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    Y_train = data['y_train']
    Y_test = data['y_test']

    Y_train = np.argmax(Y_train, axis=1)
    Y_test = np.argmax(Y_test, axis=1)

    X_tr_0 = X_train[np.where(Y_train == 0)]
    X_tr_8 = X_train[np.where(Y_train == 8)]
    X_train = np.concatenate([X_tr_0, X_tr_8])
    Y_train = np.array([0] * len(X_tr_0) + [8] * len(X_tr_8))
    X_train, Y_train = shuffle(X_train, Y_train)

    X_ts_0 = X_test[np.where(Y_test == 0)]
    X_ts_8 = X_test[np.where(Y_test == 8)]
    X_test = np.concatenate([X_ts_0, X_ts_8])
    Y_test = np.array([0] * len(X_ts_0) + [8] * len(X_ts_8))
    X_test, Y_test = shuffle(X_test, Y_test)

    X_train = np.reshape(
        X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
    X_test = np.reshape(
        X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

    return X_train, Y_train, X_test, Y_test


def show_data(X, Y, name):
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.title(name)
    plt.savefig('./output/' + name + '_res.jpg')
    plt.ion()
    plt.pause(1)
    plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_question',
                        nargs='+',
                        help='1 or 2 or both.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    args.sub_question = [int(i) for i in args.sub_question]

    X_train, Y_train, X_test, Y_test = get_data()

    if 1 in args.sub_question:

        pca = PCA(n_components=2)
        X_PCA = pca.fit_transform(X_train)
        show_data(X_PCA, Y_train, 'PCA')

        isomap = Isomap(n_components=2)
        X_Isomap = isomap.fit_transform(X_train)
        show_data(X_Isomap, Y_train, 'Isomap')

        lle = LLE(n_components=2)
        X_LLE = lle.fit_transform(X_train)
        show_data(X_LLE, Y_train, 'LLE')

        tsne = TSNE(n_components=2)
        X_TSNE = tsne.fit_transform(X_train)
        show_data(X_TSNE, Y_train, 'tSNE')

    if 2 in args.sub_question:
        f_nums = [1, 10, 20, 50, 100, 300]

        for num in f_nums:
            pca = PCA(n_components=num)
            X_PCA = pca.fit_transform(np.concatenate([X_train, X_test]))
            X_PCA_train = X_PCA[:X_train.shape[0]]
            X_PCA_test = X_PCA[X_train.shape[0]:]

            print("-" * 50)
            print("f_num:", num)

            lr = LogisticRegression(verbose=False)
            lr.fit(X_PCA_train, Y_train)
            m = lr.score(X_PCA_test, Y_test)
            mess = "From Logistic Regression: " + " Accuracy: " + str(m)
            high_light_mess(mess)

        num = X_train.shape[1]
        print("-" * 50)
        print("f_num:", num)

        lr = LogisticRegression(verbose=False, max_iter=50000)
        lr.fit(X_PCA_train, Y_train)
        m = lr.score(X_PCA_test, Y_test)
        mess = "From Logistic Regression: " + " Accuracy: " + str(m)
        high_light_mess(mess)
