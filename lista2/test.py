import os

import numpy as np
import matplotlib.pyplot as plt
from control.phaseplot import phase_plot
import control as ct
from numpy import pi
import scipy.integrate as spi
def feedback_linearization(x,t):
    return -2*(x[0]**2 + x[1])*x[0]

def ref(t):
    return np.tanh(30*t)

def u (x, t):

    return  11*x[1] + 10*(x[0] - ref(t))

last_x1_dot = 0.00
def sistema(x, t):
    return x[1], -u(x,t)

last_x = 0.00
last_t = 0.00
# print (str(u([0, 0], 0)))
# print (str(u([0, 0], 0.1)))

t = np.linspace (0, 6, 100)
vsis = spi.odeint (sistema, [0, 0], t, tfirst=False)
# vsis = []
# vsis.append([0,0])

# for i in range (len(t)):
#     dt = t[1] - t[0]
#     x1_dot, x2_dot = sistema (vsis[i], t[i])
#     vsis.append([vsis[i][0]+x1_dot*dt,vsis[i][1]+x2_dot*dt])

arr_out = np.transpose(np.array (vsis))

plt.plot (t, ref(t), 'g', label='ref')
plt.plot(t, vsis[:, 0], 'b', label='x1')
plt.plot(t, vsis[:, 1], 'r', label='x2')
#plt.plot(t, arr_out[0][:-1], 'b', label='x1')
#plt.plot(t, arr_out[1][:-1], 'r', label='x2')
plt.legend ()
plt.grid()
plt.figure()
plt.plot (t, ref(t)-vsis[:, 0], 'g', label='err')
plt.grid()
plt.show()