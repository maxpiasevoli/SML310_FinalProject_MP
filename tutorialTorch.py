import torch
from torch.autograd import Variable
import numpy as np



x_dat = np.array([3.0, 4.0, 5.0, 6.0])
y_dat = np.array([6.1, 7.9, 10.2, 11.8])


a = 1.9
b = 0.1

#pred = a * x + b
#np.sum((y - pred)**2)


def cost(x, y, a, b):
    pred = a * x + b
    print('pred ', pred)
    return torch.sum(torch.pow(y - pred, 2))


x = Variable(torch.from_numpy(x_dat), requires_grad = True)
y = Variable(torch.from_numpy(y_dat), requires_grad = True)
a = Variable(torch.from_numpy(np.array([1.9])), requires_grad = True)
b = Variable(torch.from_numpy(np.array([0.1])), requires_grad = True)

cur_cost = cost(x, y, a, b)

for i in range(1000):
    cur_cost = cost(x, y, a, b)
    cur_cost.backward()
    a.data = a.data - 0.01 * a.grad.data
    b.data = b.data - 0.01 * b.grad.data
    print(a.grad.data)
    print(b.grad.data)
    a.grad.data.zero_()
    b.grad.data.zero_()
    print(cur_cost.data.numpy())
    print(a.data.numpy()[0], b.data.numpy()[0], cur_cost.data.numpy()[0])
