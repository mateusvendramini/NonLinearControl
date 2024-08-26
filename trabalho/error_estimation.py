import os

import numpy as np
import matplotlib.pyplot as plt
from control.phaseplot import phase_plot
import control as ct
from numpy import pi
import scipy.integrate as spi
g = 9.8

def H(m1, m2, L1, L2, I1, I2, q1, q2):
    a1 = I1 +(m1*L1*L1)/4 + m2*(L1**2 + (L2**2)/4)
    a2 = I2 +(m2*L2**2)/4
    a3 = m2*L1*L2/2
    a4 = g*(m1*L1/2 + m2*L1)
    a5 = m2*g*L2/2
    return np.array([[a1+2*a3*np.cos(q2), a2+a3*np.cos(q2)],
                  [a2+a3*np.cos(q2), a2]])

def f(m1, m2, L1, L2, I1, I2, q1, q2, q1d, q2d, F1, F2):
    a1 = I1 +(m1*L1*L1)/4 + m2*(L1**2 + (L2**2)/4)
    a2 = I2 +(m2*L2**2)/4
    a3 = m2*L1*L2/2
    a4 = g*(m1*L1/2 + m2*L1)
    a5 = m2*g*L2/2

    return np.array([[-F1*q1d-a4*np.cos(q1)-a5*np.cos(q1+q2)+2*a3*np.sin(q2)*q1d*q2d+a3*np.sin(q2)*(q2d*q2d)], [-F2*q2d-a5*np.cos(q1+q2)-a3*np.sin(q2)*(q1d**2)]])

thetas = np.linspace(0, np.pi, 400)

m1h = 5
m2h = 3
L1h = 1.5
L2h = 1
I1h = 0.25
I2h = 0.125
F1h= 15
F2h=15

# m1 = 2
# m2 = 1
# L1 = 1
# L2 = 0.5
# F1 = 10
# F2 = 10
# I1 = 0.1
# I2 = 0.05

m1 = 9
m2 = 5
L1 = 2
L2 = 1.5
F1 = 20
F2 = 20
I1 = 0.4
I2 = 0.2
g = 9.8

max_error = np.array([[0], [0]])

tt1 = 0
tt2 = 0
for theta1 in thetas:
    for theta2 in thetas:
        f_hat = f(m1h, m2h, L1h, L2h, I1h, I2h, theta1, theta2, 1, 2, F1h, F2h)
        f_r = f(m1, m2, L1, L2, I1, I2, theta1, theta2, 1, 2, F1, F2)
        H_hat = H(m1h, m2h, L1h, L2h, I1h, I2h, theta1, theta2)
        H_ = H(m1, m2, L1, L2, I1, I2, theta1, theta2)
        err_ = np.matmul(np.linalg.inv(H_hat), (f_hat-f_r))
        err1 = np.matmul(H_, err_)
        err1 = np.matmul(np.linalg.inv(H_hat), err1)

        err2 = np.matmul(H_hat, err_)
        err2 = np.matmul(np.linalg.inv(H_), err2)

        a = err1[0][0] if err1[0][0] > err2[0][0] else err2[0][0]
        b = err1[1][0] if err1[1][0] > err2[1][0] else err2[1][0]
        err = np.array([[a], [b]])
        # print('theta1 {0}, theta2 {1} \r\n' .format(theta1, theta2))
        # print(err)
        # print('\r\n')
        if (abs(err[0][0]) > abs(max_error[0][0])):
            max_error[0][0] = err[0][0]
            print("max0 updated for {0}, {1}".format(theta1, theta2))
            tt1=theta1
            tt2=theta2
        if (abs(err[1][0]) > abs(max_error[1][0])):
            max_error[1][0] = err[1][0]
            print("max1 updated for {0}, {1}".format(theta1, theta2))
print("max0 updated for {0}, {1}".format(tt1, tt2))
print(max_error)

