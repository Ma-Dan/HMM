# --*--coding:utf-8 --*--
import numpy as np
import copy

def forward(observations, states, start_prob, trans_prob, emit_prob):
    alpha = np.zeros([len(observations), len(states)])

    for s in states:
        alpha[0][s] = start_prob[s] * emit_prob[s][observations[0]]

    for t in range(1, len(observations)):
        for s in states:
            prob = 0
            for s0 in states:
                prob = prob + alpha[t-1][s0] * trans_prob[s0][s]
            alpha[t][s] = prob * emit_prob[s][observations[t]]

    return alpha, sum(alpha[len(observations)-1])

def backward(observations, states, start_prob, trans_prob, emit_prob):
    beta = np.zeros([len(observations), len(states)])

    for s in states:
        beta[len(observations)-1][s] = 1

    for i in range(1, len(observations)):
        for s in states:
            prob = 0
            for s0 in states:
                prob = prob + beta[len(observations)-i][s0] * trans_prob[s][s0] * emit_prob[s0][observations[len(observations)-i]]
            beta[len(observations)-i-1][s] = prob

    ob_prob = 0
    for s in states:
        ob_prob = ob_prob + start_prob[s] * emit_prob[s][observations[0]] * beta[0][s]

    return beta, ob_prob

def rand_sum_one(count):
    #产生均匀分布和为1的数组
    r = np.random.rand(count)
    r = r / sum(r)
    return r

def calc_gamma(alpha, beta, ob_prob):
    gamma = np.zeros(alpha.shape)
    for t in range(alpha.shape[0]):
        for i in range(alpha.shape[1]):
            gamma[t][i] = alpha[t][i]*beta[t][i]/ob_prob
    return gamma

def calc_ksi(alpha, beta, ob_prob, observations, trans_prob, emit_prob):
    ksi = np.zeros([alpha.shape[0]-1, alpha.shape[1], alpha.shape[1]])
    for t in range(alpha.shape[0]-1):
        for i in range(alpha.shape[1]):
            for j in range(alpha.shape[1]):
                ksi[t][i][j] = alpha[t][i]*trans_prob[i][j]*emit_prob[j][observations[t+1]]*beta[t+1][j]/ob_prob
    return ksi

def baum_welch(observations, states, start_prob, trans_prob, emit_prob):
    # 计算前向、反向矩阵
    alpha_array = []
    beta_array = []
    ob_prob_array = []
    for d in range(len(observations)):
        alpha, ob_prob = forward(observations[d], states, start_prob, trans_prob, emit_prob)
        alpha_array.append(alpha)
        ob_prob_array.append(ob_prob)
        beta, ob_prob = backward(observations[d], states, start_prob, trans_prob, emit_prob)
        beta_array.append(beta)

    # 验算
    '''print(ob_prob_array[0])
    test = 0
    for r in range(3):
        for s in range(3):
            test = test + alpha_array[0][0][r]*trans_prob[r][s]*emit_prob[s][observations[0][1]]*beta_array[0][1][s]
    print(test)'''

    # 计算状态占有率
    gamma_array = []
    for d in range(len(observations)):
        gamma = calc_gamma(alpha_array[d], beta_array[d], ob_prob_array[d])
        gamma_array.append(gamma)

    # 计算状态转移占有率
    ksi_array = []
    for d in range(len(observations)):
        ksi = calc_ksi(alpha_array[d], beta_array[d], ob_prob_array[d], observations[d], trans_prob, emit_prob)
        ksi_array.append(ksi)

    # 计算新的参数
    new_start_prob = np.zeros(start_prob.shape)
    for d in range(len(observations)):
        new_start_prob = new_start_prob + gamma_array[d][0]
    new_start_prob = new_start_prob / len(observations)

    new_trans_prob = np.zeros(trans_prob.shape)
    for i in range(trans_prob.shape[0]):
        for j in range(trans_prob.shape[1]):
            numerator = 0
            denominator = 0
            for d in range(len(observations)):
                for t in range(len(observations[d])-1):
                    numerator = numerator + ksi_array[d][t][i][j]
                    denominator = denominator + gamma_array[d][t][i]
            new_trans_prob[i][j] = numerator / denominator

    new_emit_prob = np.zeros(emit_prob.shape)
    for i in range(emit_prob.shape[0]):
        for j in range(emit_prob.shape[1]):
            numerator = 0
            denominator = 0
            for d in range(len(observations)):
                for t in range(len(observations[d])):
                    if observations[d][t] == j:
                        numerator = numerator + gamma_array[d][t][i]
                    denominator = denominator + gamma_array[d][t][i]
            new_emit_prob[i][j] = numerator / denominator

    return new_start_prob, new_trans_prob, new_emit_prob

if __name__ == '__main__':
    # 定义隐含状态个数
    state_count = 3
    # 定义观察状态个数
    observe_count = 2

    # 隐含状态
    states = [i for i in range(state_count)]

    # 随机初始化起始状态
    start_prob = rand_sum_one(state_count)
    # 随机初始化转移矩阵
    trans_prob = np.zeros([state_count, state_count])
    for i in range(state_count):
        trans_prob[i] = rand_sum_one(state_count)
    # 随机初始化发射矩阵
    emit_prob = np.zeros([state_count, observe_count])
    for i in range(state_count):
        emit_prob[i] = rand_sum_one(observe_count)

    # 观察状态(训练数据)
    observations = [[0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1]]

    # 开始迭代
    print(emit_prob)
    for i in range(100):
        start_prob, trans_prob, emit_prob = baum_welch(observations, states, start_prob, trans_prob, emit_prob)
        print(emit_prob)