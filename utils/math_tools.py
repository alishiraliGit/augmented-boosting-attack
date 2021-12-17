import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def bits2(n, m):
    raw_bin_list = [int(x) for x in bin(n)[2:]]
    pad_bin_list = [0]*(m - len(raw_bin_list)) + raw_bin_list
    return pad_bin_list


def argmaxs(x_org, conf=0.99):
    x = x_org.copy()
    sum_x = np.sum(x)
    sum_selected = 0

    indices = np.zeros((1, x.ndim)).astype(int)
    values = np.zeros((1,))
    while sum_selected / sum_x < conf:
        indices = np.concatenate((indices, np.argwhere(x == np.max(x))[0].reshape((1, -1))), axis=0)

        val = x[tuple(indices[-1])]
        values = np.concatenate((values, [val]))

        sum_selected += val
        x[tuple(indices[-1])] = 0

    return indices[1:], values[1:]
