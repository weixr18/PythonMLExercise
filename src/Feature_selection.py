import numpy as np
import argparse
import matplotlib.pyplot as plt
from minepy import MINE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def DBCWC_select(X, y, n_f, c=2):
    """
    Select features based on Distance between class & within class.
    Use the optimal combination of individual features as the global optimal feature.
    """
    total_n_f = X.shape[1]

    Jd = np.zeros(total_n_f)
    for i in range(total_n_f):
        """For each dimensional feature"""
        X_i = X[:, i]
        m_i = np.mean(X_i)

        X_i_c = []
        m_i_c = np.zeros(c)
        S_w_c = []
        S_b_c = []
        for g in range(c):
            """For each class"""
            X_i_c[g:] = [[X_i[j] for j in range(len(X_i)) if y[j] == g]]
            m_i_c[g] = np.mean(X_i_c[g])
            S_w_c[g:] = [np.sum(np.square(X_i_c[g] - m_i_c[g]))]
            S_b_c[g:] = [len(X_i_c[g]) * np.square(m_i - m_i_c[g])]

        S_b = np.sum(S_b_c) / len(X_i)
        S_w = np.sum(S_w_c) / len(X_i)
        Jd[i] = np.log(np.abs(S_b / S_w))

    return np.argsort(Jd)[-n_f:][::-1]


def MIC_select(X, y, n_f, c=2):
    """maximal information coefficent basis"""

    total_n_f = X.shape[1]
    mine = MINE(alpha=0.6, c=15)
    y = np.reshape(y, y.shape[0])

    MIC = np.zeros(total_n_f)
    for i in range(total_n_f):
        """For each dimensional feature"""
        X_i = X[:, i]
        mine.compute_score(X_i, y)
        MIC[i] = mine.mic()

    return np.argsort(MIC)[-n_f:][::-1]


def forward_select(X, y, n_fs, c=2):
    feature_sets = []
    n_f_total = X.shape[1]
    cur_f_list = []
    all_f_set = set(range(n_f_total))

    train_n_X, valid_X, train_n_y, valid_y = train_test_split(X, y,
                                                              train_size=0.75,
                                                              )

    for i in range(max(n_fs)):
        alternative_f = list(all_f_set - set(cur_f_list))
        scores = np.zeros(len(alternative_f))
        for j, f in enumerate(alternative_f):
            f_s = list(set(cur_f_list) | {f})
            train_X_f = train_n_X[:, f_s]
            valid_X_f = valid_X[:, f_s]

            lr = LogisticRegression()
            w = lr.fit(train_X_f, train_n_y)
            acc = lr.score(valid_X_f, valid_y)

            scores[j] = acc

        sel_f = np.argmax(scores)
        cur_f_list.append(alternative_f[sel_f])
        # print("n_f:", i+1, "score:", scores[sel_f], "cur:", cur_f_list)

        if (i+1) in n_fs:
            feature_sets.append(cur_f_list.copy())

    return feature_sets


def get_txt_data(path):
    data = []
    with open(path, "r") as f:
        # heads = f.readline()
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            data.append(line)
        data = [[float(j) for j in i] for i in data]
        # heads = heads.split('\t')
        return np.array(data)
    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_question',
                        nargs='+',
                        help='select which subquestion you want me to display')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    args.sub_question = [int(i) for i in args.sub_question]

    X = get_txt_data("../data/feature_selection_X.txt")
    Y = get_txt_data("../data/feature_selection_Y.txt")
    Y = np.ravel(Y)

    class_num = np.unique(Y).size
    n_fs = [1, 5, 10, 20, 50, 100]
    train_X, test_X, train_y, test_y = train_test_split(X, Y,
                                                        train_size=300,
                                                        test_size=100,
                                                        )

    if 1 in args.sub_question:
        """DBCWC and max information basis"""

        accs_1 = []
        accs_2 = []
        f_s_1 = []
        f_s_2 = []
        for n_f in n_fs:
            f1 = DBCWC_select(train_X, train_y, n_f=n_f, c=class_num)
            f2 = MIC_select(train_X, train_y, n_f=n_f, c=class_num)

            train_X_1 = train_X[:, f1]
            test_X_1 = test_X[:, f1]

            lr = LogisticRegression(max_iter=3000)
            w1 = lr.fit(train_X_1, train_y)
            acc1 = lr.score(test_X_1, test_y)
            accs_1.append(acc1)
            f_s_1 = f1

            train_X_2 = train_X[:, f2]
            test_X_2 = test_X[:, f2]

            lr = LogisticRegression(max_iter=3000)
            w2 = lr.fit(train_X_2, train_y)
            acc2 = lr.score(test_X_2, test_y)
            accs_2.append(acc2)
            f_s_2 = f2

        print("-" * 50)
        print("from DBCWC basis:")
        print("selected features:")
        print(f_s_1)
        print("max accuracy is", max(accs_1),
              "at features", n_fs[np.argmax(accs_1)])

        print("-" * 50)
        print("from MIC basis:")
        print("selected features:")
        print(f_s_2)
        print("max accuracy is", max(accs_2),
              "at features", n_fs[np.argmax(accs_2)])

        plt.plot(n_fs, accs_1)
        plt.plot(n_fs, accs_2)
        plt.ylim([0, 1])
        plt.show()

    if 2 in args.sub_question:

        print("-" * 50)
        print("from front algorithm:")
        n_fs = list(range(1, 20))
        f_s = forward_select(train_X, train_y, n_fs)
        accs = []
        for f in f_s:
            train_X_p = train_X[:, f]
            test_X_p = test_X[:, f]

            lr = LogisticRegression()
            w = lr.fit(train_X_p, train_y)
            acc = lr.score(test_X_p, test_y)
            accs.append(acc)

        print("selected features:", f_s[-1])
        print("max accuracy is", max(accs),
              "at features", n_fs[np.argmax(accs)])
        plt.plot(n_fs, accs)
        plt.ylim([0, 1])
        plt.show()

    if 3 in args.sub_question:

        n_fs = [1, 5, 10, 20, 50, 100]
        accs = []
        accs_lin = []
        f_s = []
        for i in n_fs:

            # print("-" * 50)
            # print("number of features:", n_f)
            tree = DecisionTreeClassifier(max_features=i)
            tree.fit(train_X, train_y)
            acc = tree.score(test_X, test_y)
            accs.append(acc)

            lr = LogisticRegression(max_iter=2000)
            w1 = lr.fit(train_X, train_y)
            acc1 = lr.score(test_X, test_y)
            accs_lin.append(acc1)

            f = np.argsort(tree.feature_importances_)[-i:][::-1]
            # print("\nfeatures from decision tree:\n", f)
            f_s.append(f)

        plt.plot(n_fs, accs)
        plt.ylim([0, 1])
        plt.show()

        print('\n')
        print("-" * 50)
        print(accs_lin)
        print("-" * 50)
        print("from decision tree:")
        print("max accuracy is", max(accs),
              "at features num:", n_fs[np.argmax(accs)])
        print('selected features:', f_s[np.argmax(accs)])

    pass
