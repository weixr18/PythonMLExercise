import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import shuffle
from sklearn.metrics import silhouette_score as _SC
import matplotlib.pyplot as plt


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_question',
                        nargs='+',
                        help='1 or 2 or both.')
    args = parser.parse_args()
    return args


def get_data():
    data = np.load('./data/mnist.npz')
    X_train = data['X_train']
    Y_train = data['y_train']

    X_train = X_train / 255

    Y_train = np.argmax(Y_train, axis=1)

    labels = [0, 1, 2]
    X_t = []
    Y_t = []
    N_Samples = 300     # 每类只选300个样本

    for l in labels:
        X_tr = X_train[np.where(Y_train == l)]
        X_t.append(X_tr[:N_Samples])
        Y_tr = [l] * len(X_tr)
        Y_t.append(Y_tr[:N_Samples])
    X_train = np.concatenate(X_t)
    Y_train = np.concatenate(Y_t)

    X_train, Y_train = shuffle(X_train, Y_train)

    X_train = np.reshape(
        X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]))

    return X_train, Y_train


def J_e(X, Y, n_clusters):
    """误差平方和"""
    X_segs = []
    J = 0
    for i in range(n_clusters):
        X_i = X[np.where(Y == i)]
        m_i = np.mean(X_i, axis=0)
        J_i = np.sum(np.square(X_i - m_i))
        J += J_i

    return J


def NMI(X, Y_train, Y_pred):
    """标准化互信息"""
    n_train = np.unique(Y_train).size
    n_pred = np.unique(Y_pred).size

    P_joint = np.zeros([n_train, n_pred])
    P_train = np.zeros(n_train)
    P_pred = np.zeros(n_pred)

    for i in range(n_train):
        P_train[i] = X[np.where(Y_train == i)].shape[0]
    for j in range(n_pred):
        P_pred[j] = X[np.where(Y_pred == j)].shape[0]
        for i in range(n_train):
            a = np.where(Y_train == i)[0].tolist()
            b = np.where(Y_pred == j)[0].tolist()
            P_joint[i, j] = len(set(a) & set(b))

    P_joint /= X.shape[0]
    P_train /= X.shape[0]
    P_pred /= X.shape[0]

    nmi = 0
    for j in range(n_pred):
        for i in range(n_train):
            if P_joint[i, j] == 0:
                pass
            else:
                nmi += P_joint[i, j] * (np.log(P_joint[i, j])
                                        - np.log(P_pred[j])
                                        - np.log(P_train[i]))

    return nmi


def SC(X, Y, n_clusters):
    """轮廓系数"""
    if n_clusters <= 1:
        return 1

    X_c = []
    for i in range(n_clusters):
        X_c.append(X[np.where(Y == i)])

    def l(j, x):
        return np.sum(np.square(X_c[j] - x)) / X_c[j].shape[0]

    scs = []

    for i in range(n_clusters):
        for x in X_c[i]:
            a = np.sum(np.square(X_c[i] - x))
            a /= X_c[i].shape[0]
            b_list = [l(j, x) for j in range(n_clusters) if j != i]
            b = min(b_list)
            s = (b - a) / max(a, b)
            scs.append(s)

    sc = np.mean(scs)

    return sc


def get_cluster_mean(X, Y):
    n_clusters = np.unique(Y).size
    x_m = []
    for i in range(n_clusters):
        x_m.append(np.mean(X[np.where(Y == i)], axis=0))
    return x_m


def get_farest(X, x_m):

    dis = []
    for x in x_m:
        dis.append(np.sum(np.square(X - x), axis=1))

    dis = np.sum(np.array(dis), axis=0)
    x_far = X[np.argmax(dis)]
    return x_far


if __name__ == '__main__':
    args = get_args()
    args.sub_question = [int(i) for i in args.sub_question]
    X_train, Y_train = get_data()

    if 1 in args.sub_question:
        N_Clusters = list(range(2, 11))

        j_e = np.zeros(max(N_Clusters))
        nmi = np.zeros(max(N_Clusters))
        sc = np.zeros(max(N_Clusters))

        for n in N_Clusters:
            kmeans = KMeans(n_clusters=n)
            Y_pred = kmeans.fit_predict(X_train)
            j_e[n - 1] = J_e(X_train, Y_pred, n)
            nmi[n - 1] = NMI(X_train, Y_train, Y_pred)
            sc[n - 1] = _SC(X_train, Y_pred) if n > 1 else 0.5
            print(n, j_e[n - 1], nmi[n - 1], sc[n - 1])

        plt.subplot(1, 3, 1)
        plt.title("Minimum Square Error")
        plt.plot(N_Clusters, j_e[1:])

        plt.subplot(1, 3, 2)
        plt.title("Normalized Mutual Information")
        plt.plot(N_Clusters, nmi[1:])

        plt.subplot(1, 3, 3)
        plt.title("Silhouette Coefficient")
        plt.plot(N_Clusters, sc[1:])

        plt.show()

    if 2 in args.sub_question:

        ROUND = 10
        scores = np.zeros([ROUND, 4])

        for i in range(ROUND):

            """随机抽取法"""

            kmeans = KMeans(n_clusters=3, init='random',
                            n_init=1
                            )
            Y_pred = kmeans.fit_predict(X_train)
            s_0 = NMI(X_train, Y_train, Y_pred)
            scores[i, 0] = s_0

            """k-1聚类法"""

            # 1-means聚类相当于不聚。只寻找最远点。
            x_m_1 = np.mean(X_train, axis=0)
            a = np.sum(np.square(X_train - x_m_1), axis=1)
            x_far_1 = X_train[np.argmax(a)]

            km_2 = KMeans(n_clusters=2,
                          init=np.array([x_m_1, x_far_1]),
                          n_init=1
                          )
            Y_pred_2 = km_2.fit_predict(X_train)
            x_m_2 = get_cluster_mean(X_train, Y_pred_2)
            x_far_2 = get_farest(X_train, x_m_2)

            a = np.array(x_m_2+[x_far_2])
            km_3 = KMeans(n_clusters=3,
                          init=a,
                          n_init=1
                          )
            Y_pred_3 = km_3.fit_predict(X_train)
            s_1 = NMI(X_train, Y_train, Y_pred_3)
            scores[i, 1] = s_1

            """随机划分中心法"""

            Y_rand = np.random.randint(0, 3, size=[X_train.shape[0]])
            x_m = get_cluster_mean(X_train, Y_rand)
            kmeans = KMeans(n_clusters=3,
                            init=np.array(x_m),
                            n_init=1
                            )
            Y_pred = kmeans.fit_predict(X_train)
            s_2 = NMI(X_train, Y_train, Y_pred)
            scores[i, 2] = s_2

            """默认方法"""

            kmeans = KMeans(n_clusters=3,
                            init='k-means++',
                            n_init=1
                            )
            Y_pred = kmeans.fit_predict(X_train)
            s_3 = NMI(X_train, Y_train, Y_pred)
            scores[i, 3] = s_3

            high_light_mess(scores[i])

        high_light_mess(str(np.mean(scores, axis=0)) + "←-- mean score")

    if 3 in args.sub_question:

        metrics = ["euclidean", "l1", "l2",
                   "manhattan", "cosine"]
        linkages = ["complete", "average", "single"]

        """
        metrics = ["euclidean"]
        linkages = ["ward"]
        """

        scores = np.zeros([len(metrics), len(linkages)])

        ROUND = 10

        for i in range(ROUND):
            for j in range(len(metrics)):
                for k in range(len(linkages)):
                    agc = AgglomerativeClustering(n_clusters=3,
                                                  affinity=metrics[j],
                                                  linkage=linkages[k],
                                                  )
                    Y_pred = agc.fit_predict(X_train)
                    score = NMI(X_train, Y_train, Y_pred)
                    scores[j][k] += score

        scores /= ROUND

        for j in range(len(metrics)):
            for k in range(len(linkages)):
                print(metrics[j], linkages[k], scores[j][k])

    if 4 in args.sub_question:
        pass
