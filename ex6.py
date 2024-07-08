import os

import numpy as np
import matplotlib.pyplot as plt
from control.phaseplot import phase_plot
import control as ct
from numpy import pi
import scipy.integrate as spi



# def oscillator_ode(x, t, m=1., b=1, k=1):
#     return x[1], -k/m*x[0] - b/m*x[1]

def ex6_ode_nocontrol(x, t, m=1., b=1, k=1):
    return np.sin(x[1])*np.cos(x[0]) +x[0]**3, -x[0]*x[1]*np.cos(x[0])

def ex6_ode(x, t, m=1., b=1, k=1):
    return x[1]**2 * np.cos(x[0])-x[0], -x[0]*x[1]*np.cos (x[0])

plt.close('all')
# Generate a vector plot for the damped oscillator
# plt.figure()
# plt.clf()
# plt.axis([-3, 3, -3, 3])
# plt.title('Diagrama de fase para sistema original')
x0 = np.linspace (-3, 3, 20) 
x1 = np.linspace (-3, 3, 20)
Xex0 = []
for x in x0:
    for y in x1:
        Xex0.append([x, y])

# phase_plot(
#     ex6_ode_nocontrol,
#     X0=Xex0,
#     T=np.linspace(0, 1, 10),
#     logtime=(3, 0.7)
# )
# Generate a vector plot for the damped oscillator
plt.figure()
plt.clf()
plt.axis([-3, 3, -3, 3])
plt.title('Diagrama de fase para sistema controlado')
#print (Xex0)
# Outer trajectories
phase_plot(
    ex6_ode,
    X0=Xex0,
    T=np.linspace(0, 10, 20),
    logtime=(3, 0.7)
)

plt.figure()
plt.clf()
plt.axis([0, 10, 0, 10])
plt.title('Resposta para condição inicial (10, 1)')

t = np.linspace (0, 10, 100)
x0 = 10
y0 = 1
v = spi.odeint (ex6_ode, [x0, y0], t, tfirst=False)
plt.plot(t, v[:, 0], 'b', label='x1')
plt.plot(t, v[:, 1], 'r', label='x2')

plt.show()