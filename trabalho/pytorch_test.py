import torch
import torch.nn as nn
from torch.autograd import Variable
import torch_directml
import numpy as np
import os
from torch.utils.data import WeightedRandomSampler 
import matplotlib.pyplot as plt
def map_m1 (num):
    return map_var(num, 2, 8 )
def map_m2(num):
    return map_var(num, 1, 5)
def map_l1(num):
    return map_var(num, 1, 2)
def map_l2(num):
    return map_var(num, 0.5, 1.5)
def map_I1(num):
    return map_var(num, 0.1, 0.4)
def map_I2(num):
    return map_var(num, 0.05, 0.2)
def map_F(num):
    return map_var(num, 10, 20)


def map_var (num, min, max):
    return num*(max-min) + min

def H(m1, m2, L1, L2, I1, I2, q1, q2):
    g = 9.8
    a11 = I1
    a12 = (m1*L1*L1)/4
    a13 = m2*(L1**2 + (L2**2)/4)
    a1 = a11 + a12 + a13
    
    a2 = I2 +(m2*L2**2)/4
    a3 = m2*L1*L2/2
    return np.array([[a1 + 2*a3*np.cos(q2), a2+a3*np.cos(q2)], 
                     [a2+a3*np.cos(q2), a2]])
def f(m1, m2, L1, L2, I1, I2, q1, q2, q3, q4):
    g = 9.8
    a1 = I1 +(m1*L1*L1)/4 + m2*(L1**2 + (L2**2)/4)
    a2 = I2 +(m2*L2**2)/4
    a3 = m2*L1*L2/2
    a4 = g*(m1*L1/2 + m2*L1)
    a5 = m2*g*L2/2
    F = np.array([[F1*q3], [F2*q4]])
    C = np.array([[-2*a3*np.sin(q2)*q3*q4 - a3*np.sin(q2)*q4*q4], 
                 [q3*np.sin(q2)*q3*q3]])
    E = np.array([[a4*np.cos(q1)+a5*np.cos(q1+q2)],
                  [a5*np.cos(q1+q2)]])
    return -F-C-E


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
# dataset_size = len(a)
# validation_split = 0.2
# indices = list(range(len(a)))
# split = int(np.floor(validation_split * dataset_size))
# np.random.seed(666)
# np.random.shuffle(indices)
# train_indices = indices[split:]
# test_indices = indices[:split]
# a_test = a[test_indices]
# a_train = a[train_indices]
# #print('a_test=')
# #print(a_test)
# #print('a_train=')

# #print(a_train)
# #WeightedRandomSampler()
# train_loss = np.load(os.path.join('.', 'array_norm3', 'train_deep2.loss.npy'), 'r')
# test_loss = np.load(os.path.join('.', 'out', 'test_deep2.loss.npy'), 'r')

# plt.figure()
# plt.plot(train_loss, 'b', label='train loss')
# plt.plot(test_loss, 'r', label='test loss')
# plt.legend()
# plt.axis()

# plt.show()

print()

print('m1', map_m1(7.4951e-01)) 
print('m2', map_m2(9.0594e-01)) 
print('l1', map_l1(6.0557e-01)) 
print('l2', map_l2(1.8628e-01)) 
print('I1', map_I1(6.1144e-01)) 
print('I2', map_I2(6.4507e-01)) 
print('F1', map_F(1.4878e-01)) 
print('F2', map_F(8.7022e-02)) 


m1 = map_m1(7.4951e-01)
m2 = map_m2(9.0594e-01)
L1 = map_l1(6.0557e-01)
L2 = map_l2(1.8628e-01)
I1 = map_l1(6.1144e-01)
I2 = map_l2(6.4507e-01)
F1 = map_F(1.4878e-01)
F2 = map_F(8.7022e-02)

m1_ = map_m1( 9.6304e-01)
m2_ = map_m2( 6.6213e-02)
L1_ = map_l1( 2.3376e-01)
L2_ = map_l2( 7.7977e-02)
I1_ = map_I1( 3.8074e-01)
I2_ = map_I2( 9.2720e-01)
F1_ = map_F(2.7111e-01)
F2_ = map_F(6.6725e-01)


print('m1', m1_) 
print('m2',  m2_)
print('l1',  L1_)
print('l2',  L2_)
print('I1',  I1_)
print('I2',  I2_)
print('F1',  F1_)
print('F2',  F2_)

q1 = -4.1413e-02
q2 = 4.5732e-02
q3 = 2.7582e-01
q4=  -2.0803e-01
T1 = -6.7439e+02
T2 = -4.7262e+01

q3_=2.6087e-01
q4_= -1.7566e-01

H_i= H(m1, m2, L1, L2, I1, I2, q1, q2)
print('Hi', H_i)
H_i_= H(m1_, m2_, L1_, L2_, I1_, I2_, q1, q2)
print('Hi_', H_i_)
H_i_inv = np.linalg.inv(H_i_)
print('H_i_inv', H_i_inv)

fi = f(m1, m2, L1, L2, I1, I2, q1, q2, q3, q4)
print('fi', fi)

fi_ = f(m1_, m2_, L1_, L2_, I1_, I2_, q1, q2, q3, q4)
print('fi_', fi_)


e = np.matmul(H_i_inv, fi-fi_)
e = e - np.matmul(np.matmul(H_i_inv, H_i_), np.array([[T1], [T2]]))
dt = 0.000030517578125
print('e=', e,'\n', dt*e)
# , ->
# ,  -> 
# , -> 
# ,  -> 
# ,  -> 
# , -> 
#  -> 

