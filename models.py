import numpy as np
import torch

torch.set_default_dtype(torch.float32)


def relation_matrix_torch(vec1, vec2, e):
    dist_matrix = torch.cdist(vec1, vec2, p=1)
    if e == -1:
        relation_matrix = dist_matrix < 1e-6
        return relation_matrix.float()
    relation_matrix = 1 - dist_matrix
    over_distance = dist_matrix > e
    relation_matrix[over_distance] = 0
    return relation_matrix


def FRAD_torch(data, gamma=0.5, nominals=None, dim=None):
    n, m = data.shape
    data = torch.tensor(data, dtype=torch.float32)
    data = data.reshape(n, m, 1)

    delta = np.full(m, gamma)
    if nominals is None:
        data_min, data_max = data.min(axis=0), data.max(axis=0)
        numerical1 = np.logical_and(data_min < 1e-6, data_min > -1e-6)
        numerical2 = np.logical_and(data_max > 1-1e-6, data_max < 1+1e-6)
        nominals = np.logical_not(np.logical_and(numerical1, numerical2))
    delta[nominals] = -1


    # print("Calculating entropy of attribute...")
    Ea = torch.zeros(m, dtype=torch.float32)
    card_a = torch.zeros((m, n), dtype=torch.float32)
    for i in range(m):
        card_a[i] = relation_matrix_torch(data[:, i], data[:, i], delta[i]).sum(axis=0)
        Ea[i] = -torch.sum(torch.log2(card_a[i] / n)) / n
        # print('{:.1f}%'.format(i/m*100), end=' ')

    # print("Calculating cardinality of attribute subset...")
    if dim is None:
        feat_size = m
    else:
        feat_size = dim

    sig_a = np.array(np.argsort(Ea))
    sig_A = sig_a[::-1][:feat_size]

    card_A = torch.zeros((feat_size, n), dtype=torch.float32)
    first = sig_A[0]
    rel_matrix_pre = relation_matrix_torch(data[:, first], data[:, first], delta[first])
    card_A[0] = rel_matrix_pre.sum(axis=0)
    for i in range(1, feat_size):
        next = sig_A[i]
        rel_matrix_next = relation_matrix_torch(data[:, next], data[:, next], delta[next])
        rel_matrix_pre = np.minimum(rel_matrix_pre, rel_matrix_next)
        card_A[i] = rel_matrix_pre.sum(axis=0)
        del rel_matrix_next
        # print('{:.1f}%'.format(i / m * 100), end=' ')
    del rel_matrix_pre

    relative_dif_A = (card_A[:-1] - card_A[1:]) / card_A[:-1]
    FRAD = 1 - (card_a / n).mean(axis=0) * torch.sqrt(relative_dif_A.mean(axis=0))

    return FRAD


def FRAD_example(data, dim=100, gamma=0.5):
    n, m = data.shape
    data = torch.tensor(data, dtype=torch.float32)
    data = data.reshape(n, m, 1)

    delta = torch.zeros(m)
    for i in range(m):
        if data[:, i].max() > 1:
            delta[i] = -1
        else:
            delta[i] = gamma

    print(delta)
    # print("Calculating entropy of attribute...")
    Ea = torch.zeros(m, dtype=torch.float32)
    card_a = torch.zeros((m, n), dtype=torch.float32)
    for i in range(m):
        rel_matrix = relation_matrix_torch(data[:, i], data[:, i], delta[i])
        print(np.array(rel_matrix, dtype=np.float32))
        card_a[i] = rel_matrix.sum(axis=0)

        Ea[i] = -torch.sum(torch.log2(card_a[i] / n)) / n
        # print('{:.1f}%'.format(i/m*100), end=' ')
    print(Ea)
    # print("Calculating cardinality of attribute subset...")
    feat_size = dim
    sig_a = np.array(np.argsort(Ea))
    print('sig_a', sig_a)
    sig_A = sig_a[::-1][:feat_size]
    print('sig_A', sig_A)
    card_A = torch.zeros((feat_size, n), dtype=torch.float32)
    first = sig_A[0]
    rel_matrix_pre = relation_matrix_torch(data[:, first], data[:, first], delta[first])
    card_A[0] = rel_matrix_pre.sum(axis=0)
    print(rel_matrix_pre)
    for i in range(1, feat_size):
        next = sig_A[i]
        rel_matrix_next = relation_matrix_torch(data[:, next], data[:, next], delta[next])
        rel_matrix_pre = np.minimum(rel_matrix_pre, rel_matrix_next)
        print(rel_matrix_pre)
        card_A[i] = rel_matrix_pre.sum(axis=0)
        del rel_matrix_next
        # print('{:.1f}%'.format(i / m * 100), end=' ')
    del rel_matrix_pre
    print(card_A)
    relative_dif_A = (card_A[:-1] - card_A[1:]) / card_A[:-1]
    print(relative_dif_A)

    FRAD = 1 - (torch.sqrt(card_a / n)).mean(dim=0) * relative_dif_A.mean(axis=0)
    return FRAD


if __name__ == "__main__":
    pass
