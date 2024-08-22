import os

import numpy as np
import matplotlib.pyplot as plt
from control.phaseplot import phase_plot
import control as ct
from numpy import pi
import scipy.integrate as spi

def u (x):
    return -np.cos(x[0]) -3*x[0] + x[1]+x[0]-7*x[1]
def sistema(x, t):
    return -x[0] + 7*x[1], -x[1] + np.cos(x[0]) + u(x)


t = np.linspace (0, 10, 1000)
vsis = spi.odeint (sistema, [5, 5], t, tfirst=False)
plt.plot(t, vsis[:,0], 'b', label='x1')
plt.plot(t, vsis[:,1], 'r', label='x2')
plt.legend ()
#plt.show()

plt.figure()
plt.clf()
plt.axis([-10, 10, -10, 10])
plt.title('diagrama de fase')
x0 = np.linspace (-10, 10, 20) 
x1 = np.linspace (-10, 10, 20)
Xex0 = []
for x in x0:
    for y in x1:
        Xex0.append([x, y])

#print (Xex0)
# Outer trajectories
phase_plot(
    sistema,
    X0=Xex0, #=[[-1.0, -1.0], [1, 1], [0.1, 0.1]],
    T=np.linspace(0, 10, 1000),
    logtime=(3, 0.7)
)
# Separatrices
plt.show()