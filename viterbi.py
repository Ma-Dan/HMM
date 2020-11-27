# --*--coding:utf-8 --*--
import numpy as np

def viterbi(obs, states, start_prob, trans_prob, emit_prob):
    P = np.zeros([len(obs), len(states)])
    path = np.zeros([len(states), len(obs)])

    for s in states:
        P[0][s] = start_prob[s] * emit_prob[s][obs[0]]
        path[s][0] = s

    for t in range(1, len(obs)):
        newpath = np.zeros([len(states), len(obs)])
        for s in states:
            prob = -1
            for s0 in states:
                nprob = P[t - 1][s0] * trans_prob[s0][s] * emit_prob[s][obs[t]]
                if nprob > prob:
                    prob = nprob
                    P[t][s] = prob
                    newpath[s][0:t] = path[s0][0:t]
                    newpath[s][t] = s
        path = newpath

    prob = -1
    state = 0
    for s in states:
        if P[len(obs)-1][s] > prob:
            prob = P[len(obs)-1][s]
            state = s

    print(P)

    return path[state]


if __name__ == '__main__':
    # 隐含状态
    states = [0, 1, 2]
    # 初始状态
    start_prob = [0.63, 0.17, 0.20]
    # 转移矩阵
    trans_prob = [[0.5, 0.375, 0.125], [0.25, 0.125, 0.625], [0.25, 0.375, 0.375]]
    # 发射矩阵
    emit_prob = [[0.6, 0.2, 0.15, 0.05], [0.25, 0.25, 0.25, 0.25], [0.05, 0.10, 0.35, 0.5]]
    # 观察状态
    obs = [0, 2, 3]

    print(viterbi(obs, states, start_prob, trans_prob, emit_prob))
