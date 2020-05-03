import numpy as np
from hmmlearn import hmm

states = ['cheat', 'clear']
observations = [1, 2, 3, 4, 5, 6]

n_states = len(states)
n_observations = len(observations)


def load_data():
    data = np.load('../data/sequences.npy')
    data = data.reshape([data.size, 1])
    data -= 1
    return data


def forward_algo(A, E, Pi, O, n_states, n_observations):
    alpha = np.zeros([len(O), n_states])
    for t in range(len(O)):
        j = O[t]
        if t == 0:
            for i in range(n_states):
                alpha[t, i] = E[i, j] * Pi[i]
                print("alpha[%d, %d] = %.6f" % (t, i, alpha[t, i]))
        else:
            for i in range(n_states):
                alpha[t, i] = E[i, j] * np.dot(alpha[t - 1, :], A[:, i])
                print("alpha[%d, %d] = %.6f" % (t, i, alpha[t, i]))

    print("total probability:", np.sum(alpha[-1, :]))


def backward_algo(A, E, Pi, O, n_states, n_observations):
    beta = np.zeros([len(O), n_states])
    for t in range(len(O)-1, -1, -1):
        if t == len(O) - 1:
            for i in range(n_states):
                beta[t, i] = 1
                print("beta[%d, %d] = %.6f" % (t, i, beta[t, i]))
        else:
            j = O[t+1]
            for i in range(n_states):
                beta[t, i] = E[i, j] * np.dot(beta[t + 1, :], A[i, :])
                print("beta[%d, %d] = %.6f" % (t, i, beta[t, i]))

    p0 = [Pi[i] * beta[0, i] * E[i, O[0]] for i in range(n_states)]
    print("p0:", p0)
    print("total probability:", np.sum(np.array(p0)))


if __name__ == "__main__":

    # 读入数据，准备模型
    data = load_data()
    model = hmm.MultinomialHMM(n_components=n_states,
                               tol=1e-4,
                               n_iter=1000,
                               verbose=False,
                               )

    # 训练模型
    print("Train Starts")
    print('--'*30)
    model.fit(data, lengths=[30] * 200)

    Pi = model.startprob_
    A = model.transmat_
    E = model.emissionprob_
    print("Pi:", Pi, '\n')
    print("A:", A, '\n')
    print("E:", E, '\n')

    # 进行预测
    print("\nPrediction")
    print('--'*30)
    O = np.array([6, 6, 6, 6])
    O -= 1

    print("\nForward algorithm:")
    forward_algo(A, E, Pi, O, n_states, n_observations)  # 前向算法

    print("\nBackward algorithm:")
    backward_algo(A, E, Pi, O, n_states, n_observations)    # 后向算法

    # 进行解码
    print("\nDecode:")
    print('--'*30)
    seen = np.array([3, 2, 1, 3, 4, 5, 6, 3, 1, 4, 1, 6, 6, 2, 6])
    seen -= 1
    logprob, hidden = model.decode(seen.reshape(-1, 1), algorithm='viterbi')
    print("seen:", seen + 1)
    print("hidden:", hidden)
    print("logprob", logprob)
