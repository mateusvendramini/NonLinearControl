import os

import numpy as np
import matplotlib.pyplot as plt
from control.phaseplot import phase_plot
import control as ct
from numpy import pi
import scipy.integrate as spi

def ref(t):
    return 7-7*np.tanh(2*t)

def ref2(t):
    return np.tanh(2*t)

def convert(x1, x2):
    return x1, -x1 + 7*x2

def u (x, t):
    z1, z2 = convert (x[0], x[1])
    return  (1/7)*(2*z2 + z1 -7*np.cos(z1)+ (ref(t)-z2))

def u2 (x, t):
    z1, z2 = convert (x[0], x[1])
    return  (1/7)*(2*z2 + z1 -7*np.cos(z1)+ 10*(ref2(t)-z1) - 11*z2)

def sistema(x, t):
    return -x[0]+7*x[1], -x[1] + np.cos(x[0]) + u(x, t)


def sistema2(x, t):
    return -x[0]+7*x[1], -x[1] + np.cos(x[0]) + u2(x, t)

t = np.linspace (0, 10, 100)
vsis = spi.odeint (sistema, [0, 1], t, tfirst=False)

plt.plot(t, vsis[:,0], 'b', label='x1')
plt.plot(t, vsis[:,1], 'r', label='x2')
#print("u[0, 1] - " + str(u([0,1], 0)))
u_sig = []
for i in range(len(t)):
    u_sig.append (u(vsis[i], t[i]))

#plt.plot(t, u_sig, 'g', label='u')
plt.grid()
plt.legend ()

plt.figure()
#z1, z2 = convert(arr_out[0][:-1], arr_out[1][:-1])
z1, z2 = convert(vsis[:,0], vsis[:,1])
plt.plot(t, z1, 'b', label='z1')
plt.plot(t, z2, 'r', label='z2')
plt.plot (t, ref(t), 'g', label='z2 ref')
plt.grid()
plt.legend ()

vsis = spi.odeint (sistema2, [0, 0], t, tfirst=False)
plt.figure()
plt.plot(t, vsis[:,0], 'b', label='x1')
plt.plot(t, vsis[:,1], 'r', label='x2')
u_sig = []
for i in range(len(t)):
    u_sig.append (u(vsis[i], t[i]))

plt.plot(t, u_sig, 'g', label='u')
plt.grid()
plt.legend ()

plt.figure()
z1, z2 = convert(vsis[:,0], vsis[:,1])
plt.plot(t, z1, 'b', label='z1')
plt.plot(t, z2, 'r', label='z2')
plt.plot (t, ref2(t), 'g', label='z1 ref')
plt.grid()
plt.legend ()

plt.show()