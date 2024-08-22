import os

import numpy as np
import matplotlib.pyplot as plt
from control.phaseplot import phase_plot
import control as ct
from numpy import pi
import scipy.integrate as spi

def ref(t):
    return 0*np.tanh(10*t)

def convert(x1, x2):
    return x1, -x1 + 7*x2
def v (x, t, y_dot):
    return 11*y_dot + 10*(x[0] - ref(t))

def u (x, t, y_dot):
    return  x[2] - x[0]**4 + x[0]**2+8*x[1]-8*x[2] - v(x, t, y_dot)

def sistema(x, t):
    y_dot =  x[2] + 8*x[1]
    return x[2] + 8*x[1], -x[1] + x[2], -x[2] + x[0]**4- x[0]**2 + u(x, t, y_dot)

last_x = 0.00
last_t = 0.00
# print (str(u([0, 0], 0)))
# print (str(u([0, 0], 0.1)))

t = np.linspace (0, 5, 500)
#vsis = spi.odeint (sistema, [0, 0], t, tfirst=False)
vsis = []
vsis.append([5,-3,40])

for i in range (len(t)):
    dt = t[1] - t[0]
    x1_dot, x2_dot, x3_dot = sistema (vsis[i], t[i])
    vsis.append([vsis[i][0]+x1_dot*dt,vsis[i][1]+x2_dot*dt, vsis[i][2]+x3_dot*dt])

arr_out = np.transpose(np.array (vsis))
plt.title ("Condição inicial [%d, %d, %d] e  ref %f" %(vsis[0][0], vsis[0][1], vsis[0][2], ref(10)))
plt.plot (t, ref(t), 'g', label='ref')
plt.plot(t, arr_out[0][:-1], 'b', label='x1')
plt.plot(t, arr_out[1][:-1], 'r', label='x2')
plt.plot(t, arr_out[2][:-1], 'y', label='x3')
plt.grid()
plt.legend ()
plt.figure()

t = np.linspace (0, 5, 500)
#vsis = spi.odeint (sistema, [0, 0], t, tfirst=False)
vsis = []
vsis.append([-5,3, 0])

for i in range (len(t)):
    dt = t[1] - t[0]
    x1_dot, x2_dot, x3_dot = sistema (vsis[i], t[i])
    vsis.append([vsis[i][0]+x1_dot*dt,vsis[i][1]+x2_dot*dt, vsis[i][2]+x3_dot*dt])

arr_out = np.transpose(np.array (vsis))
plt.title ("Condição inicial [%d, %d, %d] e  ref %f" %(vsis[0][0], vsis[0][1], vsis[0][2], ref(10)))
plt.plot (t, ref(t), 'g', label='ref')
plt.plot(t, arr_out[0][:-1], 'b', label='x1')
plt.plot(t, arr_out[1][:-1], 'r', label='x2')
plt.plot(t, arr_out[2][:-1], 'y', label='x3')
plt.grid()
plt.legend ()

plt.figure()

t = np.linspace (0, 5, 500)
#vsis = spi.odeint (sistema, [0, 0], t, tfirst=False)
vsis = []
vsis.append([1, 5, -10])

for i in range (len(t)):
    dt = t[1] - t[0]
    x1_dot, x2_dot, x3_dot = sistema (vsis[i], t[i])
    vsis.append([vsis[i][0]+x1_dot*dt,vsis[i][1]+x2_dot*dt, vsis[i][2]+x3_dot*dt])

arr_out = np.transpose(np.array (vsis))
plt.title ("Condição inicial [%d, %d, %d] e  ref %f" %(vsis[0][0], vsis[0][1], vsis[0][2], ref(10)))
plt.plot (t, ref(t), 'g', label='ref')
plt.plot(t, arr_out[0][:-1], 'b', label='x1')
plt.plot(t, arr_out[1][:-1], 'r', label='x2')
plt.plot(t, arr_out[2][:-1], 'y', label='x3')
plt.grid()
plt.legend ()
plt.show()