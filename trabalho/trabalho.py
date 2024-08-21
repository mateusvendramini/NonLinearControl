import os

import numpy as np
import matplotlib.pyplot as plt
from control.phaseplot import phase_plot
import control as ct
from numpy import pi
import scipy.integrate as spi

def ref(t):
    return 1*np.tanh(10*t)

def u (x, t ):
    return 1
def p():
    return 0.5

def sistema(x, t):
    return x[1], -0.2*x[1]-4.905*np.sin(x[0])+0.1*u(x, t)+0.1*p()*x[0]*np.sin(2*x[1]*x[1])

last_x = 0.00
last_t = 0.00
# print (str(u([0, 0], 0)))
# print (str(u([0, 0], 0.1)))

t = np.linspace (0, 30, 6000)
#vsis = spi.odeint (sistema, [0, 0], t, tfirst=False)
vsis = []
vsis.append([0, 0])

for i in range (len(t)):
    dt = t[1] - t[0]
    x1_dot, x2_dot = sistema (vsis[i], t[i])
    vsis.append([vsis[i][0]+x1_dot*dt,vsis[i][1]+x2_dot*dt])

arr_out = np.transpose(np.array (vsis))
plt.title ("Condição inicial [%d, %d] u constante  [%d]" %(vsis[0][0], vsis[0][1], u([0, 0], 0)))
#plt.plot (t, ref(t), 'g', label='ref')
plt.plot(t, arr_out[0][:-1], 'b', label='x1')
plt.plot(t, arr_out[1][:-1], 'r', label='x2')
plt.grid()
plt.legend ()

plt.show()