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

def ref1(t):
    return np.pi/6

def ref2(t):
    return np.pi/6

def sat(x):
    if x>=1:
        return 1
    elif x<= -1:
        return -1
    return x
def s(x0, x1, t, lamb, ref):
    return x1+lamb*(x0-ref(t))

print(s(0, -2, 0, 4, ref2))