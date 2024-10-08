import os

import numpy as np
import matplotlib.pyplot as plt
from control.phaseplot import phase_plot
import control as ct
from numpy import pi
import scipy.integrate as spi

def ref1(t):
    return 1.127572188422499+(np.pi/2)*np.tanh(10*t)#(np.pi/2)*np.tanh(10*t) + np.pi/2

def ref2(t):
    return -0.02717109077810674-(np.pi/2)*np.tanh(10*t) #np.pi/2 - (np.pi/2) * np.tanh(10*t)

def sat(x):
    if x>=1:
        return 1
    elif x<= -1:
        return -1
    return x
def s(x0, x1, t, lamb, ref):
    return x1+lamb*(x0-ref(t))

def u (x, t):
    #return 0,0
    # m1 = 5
    # m2 = 3
    # L1 = 1.5
    # L2 = 1
    # I1 = 0.25
    # I2 = 0.125
    # F1= 15
    # F2=15
    m1 = 1.9178533132322388
    m2 = 0.9651596624103864
    L1 = 1.0233902423594152
    L2 = 0.4404871475561963
    I1 = 0.1036096982291639
    I2 = 0.05330792550109102
    F1 = 15
    F2 = 15
    g = 9.8

    a1 = I1 +(m1*L1*L1)/4 + m2*(L1**2 + (L2**2)/4)
    a2 = I2 +(m2*L2**2)/4
    a3 = m2*L1*L2/2
    a4 = g*(m1*L1/2 + m2*L1)
    a5 = m2*g*L2/2

    q1 = x[0]
    q2 = x[1]
    q1d = x[2]
    q2d = x[3]
    lamb = 8
    #phi = 0.005
    phi = 0.005

    
    q2dd_den = F2*q2d+a5*np.cos(q1+q2)+a3*np.sin(q2)*(q1d**2)
    q1dd_den = F1*q1d+a4*np.cos(q1)+a5*np.cos(q1+q2)+2*a3*np.sin(q2)*q1d*q2d+a3*np.sin(q2)*(q2d*q2d)

    H = np.array([[a1+2*a3*np.cos(q2), a2+a3*np.cos(q2)],
                  [a2+a3*np.cos(q2), a2]])
    sden = np.array([[74*sat(s(q1, q1d, t, lamb, ref1)/phi)+lamb*q1d], [266*sat(s(q2, q2d, t, lamb, ref2)/phi)+lamb*q2d]])

    slin = np.matmul(H, sden)
    #s = np.array([[s(q1, q1d, t, lamb)], [s(q2, q2d, t, lamb)]])

    U = q1dd_den-slin[0][0], q2dd_den-slin[1][0]-lamb*q2d
    return U[0], U[1]
    #return 0.8*p1-50*(x[0]-ref(t)), 0.8*p2-50*(x[1]-ref(t)) 

def p():
    return 0.5

def sistema(x, t, I1, I2, L1, L2, m1, m2, F1, F2, g):
    '''Estados q1, q2, q1dot, q2dot
    '''
    a1 = I1 +(m1*L1*L1)/4 + m2*(L1**2 + (L2**2)/4)
    a2 = I2 +(m2*L2**2)/4
    a3 = m2*L1*L2/2
    a4 = g*(m1*L1/2 + m2*L1)
    a5 = m2*g*L2/2
    q1=x[0]
    q2=x[1]
    q1d=x[2]
    q2d=x[3]
    U = u(x, t, a4*np.cos(q1)+a5*np.cos(q1+q2), a5*np.cos(q1+q2))

    q2dd_den = U[1]-F2*q2d-a5*np.cos(q1+q2)-a3*np.sin(q2)*(q1d**2)
    q1dd_den = U[0]-F1*q1d-a4*np.cos(q1)-a5*np.cos(q1+q2)+2*a3*np.sin(q2)*q1d*q2d+a3*np.sin(q2)*q2d**2
    q2dd = (q2dd_den-q1dd_den*(a2+a3*np.cos(q2)/(a1+2*a3*np.cos(q2))))/(a2-(a2+a3*np.cos(q2)*(a2+a3*np.cos(q2))/(a1+2*a3*np.cos(q2))))
    q1dd = (q1dd_den-(a2+a3*np.cos(q2))*q2dd)/(a1+2*a3*np.cos(q2))
    return q1d, q2d, q1dd, q2dd

def sistema2(x, t, I1, I2, L1, L2, m1, m2, F1, F2, g):
    '''Estados q1, q2, q1dot, q2dot
    '''
    a1 = I1 +(m1*L1*L1)/4 + m2*(L1**2 + (L2**2)/4)
    a2 = I2 +(m2*L2**2)/4
    a3 = m2*L1*L2/2
    a4 = g*(m1*L1/2 + m2*L1)
    a5 = m2*g*L2/2
    q1=x[0]
    q2=x[1]
    q1d=x[2]
    q2d=x[3]
    U = u(x, t)

    q2dd_den = U[1]-F2*q2d-a5*np.cos(q1+q2)-a3*np.sin(q2)*(q1d**2)
    q1dd_den = U[0]-F1*q1d-a4*np.cos(q1)-a5*np.cos(q1+q2)+2*a3*np.sin(q2)*q1d*q2d+a3*np.sin(q2)*(q2d*q2d)
    H = np.array([[a1+2*a3*np.cos(q2), a2+a3*np.cos(q2)],
                  [a2+a3*np.cos(q2), a2]])
    H_inv = np.linalg.inv(H)
    qdd = np.matmul(H_inv, np.array([[q1dd_den], [q2dd_den]]))
    return q1d, q2d, qdd[0][0], qdd[1][0]

# print (str(u([0, 0], 0)))
# print (str(u([0, 0], 0.1)))

t = np.linspace (0, 1, 32768)
#vsis = spi.odeint (sistema, [0, 0], t, tfirst=False)
vsis = []
vsis.append([1.127572188422499, -0.02717109077810674, 0, 0])
# m1 = 4
# m2 = 1
# L1 = 2
# L2 = 1
# F1 = 10
# F2 = 10
# I1 = 0.2
# I2 = 0.05
# m1 = 5
# m2 = 3
# L1 = 1.5
# L2 = 1
# I1 = 0.25
# I2 = 0.125
# F1= 15
# F2=15

# m1 = 9
# m2 = 5
# L1 = 2
# L2 = 1.5
# F1 = 10
# F2 = 10
# I1 = 0.4
# I2 = 0.2
# g = 9.8

m1 = 2.0975561637446507
m2 = 0.986390211428680
L1 = 0.9165574998344682
L2 = 0.49447895383476115
I1 = 0.10338316430814988
I2 = 0.04012892953284832
F1= 10
F2=10

g = 9.8
for i in range (len(t)):
    dt = t[1] - t[0]
    x1_dot, x2_dot, x3_dot, x4_dot = sistema2(vsis[i], t[i], I1, I2, L1, L2, m1, m2, F1, F2, g)
    vsis.append([vsis[i][0]+x1_dot*dt,vsis[i][1]+x2_dot*dt, vsis[i][2]+dt*x3_dot, vsis[i][3]+dt*x4_dot])
    # if (abs(vsis[i][0]) > 5 or abs(vsis[i][1])> 5):
    #     print("break loop after (%f %f) on step %d" %(vsis[i][0], vsis[i][1], i))
    #     del(vsis[-1])
    #     break

arr_out = np.transpose(np.array (vsis))
#plt.title ("Condição inicial [%d, %d] u constante  [%d]" %(vsis[0][0], vsis[0][1], u([0, 0], 0)))
#plt.plotsubplot (t, ref(t), 'g', label='ref')
plt.subplot(2, 2, 1)
plt.suptitle('Simulação controlador de modos deslizantes, grande erro parâmetros')
plt.plot(t[:len(arr_out[0])-1], ref1(t[:len(arr_out[0])-1]), 'g', label='refq1')
plt.plot(t[:len(arr_out[0])-1], ref2(t[:len(arr_out[0])-1]), 'y', label='refq2')
plt.plot(t[:len(arr_out[0])-1], arr_out[0][:-1], 'b', label='q1')
plt.plot(t[:len(arr_out[0])-1], arr_out[1][:-1], 'r', label='q2')
plt.grid()
plt.legend ()

plt.subplot(2, 2, 2)
plt.plot(t[:len(arr_out[0])-1], arr_out[2][:-1], 'b', label='q1d')
plt.plot(t[:len(arr_out[0])-1], arr_out[3][:-1], 'r', label='q2d')
plt.grid()
plt.legend ()

plt.subplot(2, 2, 3)
s1 = s(arr_out[0][:-1], arr_out[2][:-1], t, 8, ref1)
plt.plot(t, s1, 'b', label='s1')
plt.grid()
plt.legend ()

plt.subplot(2, 2, 4)
s1 = s(arr_out[1][:-1], arr_out[3][:-1], t, 8, ref2)
plt.plot(t, s1, 'r', label='s2')
plt.grid()
plt.legend ()

plt.show()