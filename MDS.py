import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import argparse


class MyMDS():
    def __init__(self, d=2):
        self.d = d
        pass

    def fit(self, D):
        n, k = D.shape
        dist = np.zeros((n, n))
        disti = np.zeros(n)
        distj = np.zeros(n)
        B = np.zeros((n, n))

        for i in range(n):
            dist[i] = np.sum(np.square(D[i]-D), axis=1).reshape(1, n)
        for i in range(n):
            disti[i] = np.mean(dist[i, :])  # 行平方平均
            distj[i] = np.mean(dist[:, i])  # 列平方平均
        distij = np.mean(dist)  # 全矩阵平方平均

        for i in range(n):
            for j in range(n):
                B[i, j] = -0.5 * (dist[i, j] - disti[i] - distj[j] + distij)

        lamda, V = np.linalg.eigh(B)  # 特征分解

        index = np.argsort(-lamda)[:self.d]
        lamda_selected = -np.sort(-lamda)
        # print(lamda_selected)
        diag_lamda = np.sqrt(np.diag(lamda[index]))
        V_selected = V[:, index]    # 只选择前d个特征向量
        self.A = V_selected.dot(diag_lamda)
        pass


def get_data():
    path = './data/city_dist.xlsx'
    data = pd.read_excel(path)
    names = data.values[:, 0]
    data = data.values[:, 1:]
    return names, data


def show_res(A, names):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(A[:, 0], A[:, 1])
    for i in range(len(A)):
        plt.annotate(names[i], xy=(A[i, 0], A[i, 1]), xytext=(
            A[i, 0] + 0.1, A[i, 1] + 0.1))
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',
                        nargs='+',
                        help='1 or 2 or both, 1 for hand-make algorithm\
                             and 2 for sklearn algorithm')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    args.method = [int(i) for i in args.method]
    names, data = get_data()

    if 1 in args.method:
        mds = MyMDS(d=2)
        mds.fit(data)
        A = mds.A
        A[:, 0:1] = -A[:, 0:1]
        # show_res(A, names)
        data_ = np.zeros((len(data), len(data)))
        for i in range(len(data)):
            for j in range(len(data)):
                data_[i, j] = np.sqrt(np.sum(np.square(A[i] - A[j])))
        err = np.sum(np.square(data - data_)) / \
            np.square(len(data))  # 估计的A的距离矩阵和原距离矩阵的均方误差

    if 2 in args.method:
        mds2 = MDS(2, dissimilarity='precomputed')
        mds2.fit(data)
        A2 = mds2.fit_transform(data)
        show_res(A2, names)

    pass
