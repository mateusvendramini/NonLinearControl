import torch
import torch.nn as nn
from torch.autograd import Variable
import torch_directml
import numpy as np
import os
from torch.utils.data import WeightedRandomSampler 
import matplotlib.pyplot as plt


a = torch.tensor([[1, 2, 3, 4],
                  [3, 4, 5, 6],
                  [7, 8, 9, 10],
                  [1, 2, 3, 4],
                  [3, 4, 5, 6],
                  [7, 8, 9, 10],
                  [1, 2, 3, 4],
                  [3, 4, 5, 6],
                  [7, 8, 9, 10],
                  [1, 2, 3, 4],
                  [3, 4, 5, 6],
                  [7, 8, 9, 10],
                  [1, 2, 3, 4],
                  [3, 4, 5, 6],
                  [7, 8, 9, 10],
                  ])
print (a)
print(torch.index_select(a, 1, torch.tensor([1])))
#a11 = torch.index_select(a, 1, torch.tensor([0]))
#a12 = torch.index_select(a, 1, torch.tensor([1]))
#a22 = torch.index_select(a, 1, torch.tensor([3]))
#a21 = torch.index_select(a, 1, torch.tensor([2]))

#print(torch.cat((a11, a12, a21, a22), 1))
dataset_size = len(a)
validation_split = 0.2
indices = list(range(len(a)))
split = int(np.floor(validation_split * dataset_size))
np.random.seed(666)
np.random.shuffle(indices)
train_indices = indices[split:]
test_indices = indices[:split]
a_test = a[test_indices]
a_train = a[train_indices]
#print('a_test=')
#print(a_test)
#print('a_train=')

#print(a_train)
#WeightedRandomSampler()
train_loss = np.load(os.path.join('.', 'array_norm3', 'train_deep2.loss.npy'), 'r')
test_loss = np.load(os.path.join('.', 'out', 'test_deep2.loss.npy'), 'r')

plt.figure()
plt.plot(train_loss, 'b', label='train loss')
plt.plot(test_loss, 'r', label='test loss')
plt.legend()
plt.axis()

plt.show()

