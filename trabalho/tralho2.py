import os

import numpy as np
import matplotlib.pyplot as plt
from control.phaseplot import phase_plot
import control as ct
from numpy import pi
import scipy.integrate as spi

def ref(t):
    return (np.pi/6)*np.tanh(2*t)

def u (x, t, p1, p2):
    #return 0,0
    return 0.8*p1-50*(x[0]-ref(t)), 0.8*p2-50*(x[1]-ref(t)) 

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
    U = u(x, t, a4*np.cos(q1)+a5*np.cos(q1+q2), a5*np.cos(q1+q2))
    U = u(x, t, a4*np.cos(q1)+a5*np.cos(q1+q2), a5*np.cos(q1+q2))

    q2dd_den = U[1]-F2*q2d-a5*np.cos(q1+q2)-a3*np.sin(q2)*(q1d**2)
    q1dd_den = U[0]-F1*q1d-a4*np.cos(q1)-a5*np.cos(q1+q2)+2*a3*np.sin(q2)*q1d*q2d+a3*np.sin(q2)*(q2d*q2d)
    H = np.array([[a1+2*a3*np.cos(q2), a2+a3*np.cos(q2)],
                  [a2+a3*np.cos(q2), a2]])
    H_inv = np.linalg.inv(H)
    qdd = np.matmul(H_inv, np.array([[q1dd_den], [q2dd_den]]))
    return q1d, q2d, qdd[0][0], qdd[1][0]

last_x = 0.00
last_t = 0.00
# print (str(u([0, 0], 0)))
# print (str(u([0, 0], 0.1)))

t = np.linspace (0, 15, 5000)
#vsis = spi.odeint (sistema, [0, 0], t, tfirst=False)
vsis = []
vsis.append([0, 0, 0, 0])
m1 = 4
m2 = 1
L1 = 2
L2 = 1
F1 = 10
F2 = 10
I1 = 0.2
I2 = 0.05
g = 9.8
for i in range (len(t)):
    dt = t[1] - t[0]
    x1_dot, x2_dot, x3_dot, x4_dot = sistema2(vsis[i], t[i], I1, I2, L1, L2, m1, m2, F1, F2, g)
    vsis.append([vsis[i][0]+x1_dot*dt,vsis[i][1]+x2_dot*dt, vsis[i][2]+dt*x3_dot, vsis[i][3]+dt*x4_dot])
    if (abs(vsis[i][0]) > 5 or abs(vsis[i][1])> 5):
        print("break loop after (%f %f) on step %d" %(vsis[i][0], vsis[i][1], i))
        del(vsis[-1])
        break

arr_out = np.transpose(np.array (vsis))
#plt.title ("Condição inicial [%d, %d] u constante  [%d]" %(vsis[0][0], vsis[0][1], u([0, 0], 0)))
#plt.plotsubplot (t, ref(t), 'g', label='ref')
plt.subplot(1, 2, 1)
plt.suptitle('Simulação em malha fechada para CI [0,0,0,0] Kp=50 erro de 20% no feedback linearization')
plt.plot(t[:len(arr_out[0])-1], ref(t[:len(arr_out[0])-1]), 'g', label='ref')
plt.plot(t[:len(arr_out[0])-1], arr_out[0][:-1], 'b', label='q1')
plt.plot(t[:len(arr_out[0])-1], arr_out[1][:-1], 'r', label='q2')
plt.grid()
plt.legend ()

plt.subplot(1, 2, 2)
plt.plot(t[:len(arr_out[0])-1], arr_out[2][:-1], 'b', label='q1d')
plt.plot(t[:len(arr_out[0])-1], arr_out[3][:-1], 'r', label='q2d')
plt.grid()
plt.legend ()

plt.show()