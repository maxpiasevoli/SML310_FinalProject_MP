import torch
from torch.autograd import Variable
import numpy as np


def _cost(x, y, a, b):
    z = a * x + b
    ind_costs = y * torch.sigmoid(z) + (1 - y) * (1 - torch.sigmoid(z))
    return torch.sum(ind_costs)

def sig(x):
    return 1/(1 + np.exp(-x))

def calcAvgLogLike(x_real_scores, x_fake_scores, y_dat):

    alpha = 0.01

    x_dat = np.vstack((x_real_scores, x_fake_scores))
    x_dat = x_dat.astype('float64')
    n_samples = x_dat.shape[0]

    x = Variable(torch.from_numpy(x_dat), requires_grad = True)
    y = Variable(torch.from_numpy(y_dat), requires_grad = True)
    a = Variable(torch.from_numpy(np.array([0.01])), requires_grad = True)
    b = Variable(torch.from_numpy(np.array([0.05])), requires_grad = True)

    # print(x.type())
    # print(y.type())
    # print(a.type())
    # print(b.type())

    # cur_cost = _cost(x, y, a, b)

    print('converting critic scores to log probability')
    for i in range(1000):
        cur_cost = _cost(x, y, a, b)
        cur_cost.backward()
        a.data = a.data - alpha * a.grad.data
        b.data = b.data - alpha * b.grad.data
        print(a.grad.data)
        print(b.grad.data)
        a.grad.data.zero_()
        b.grad.data.zero_()
        print(a.data.numpy()[0], b.data.numpy()[0], cur_cost.data.numpy())

    a = a.data.numpy()[0]
    b = b.data.numpy()[0]

    probs = sig(a * x_real_scores + b)
    print(sum(np.where(probs == 0, 1, 0)))
    print(sum(np.where(1-probs == 0, 1, 0)))
    print(x_real_scores)
    print('-------------')
    print(probs)
    print('-------------')
    print(np.log(probs))
    sum_log_like = np.sum(np.log(probs))
    return sum_log_like / x_real_scores.shape[0]
