import os

import numpy as np
import matplotlib.pyplot as plt
from control.phaseplot import phase_plot
import control as ct
from numpy import pi


# def oscillator_ode(x, t, m=1., b=1, k=1):
#     return x[1], -k/m*x[0] - b/m*x[1]

mi = 0.1
def ex3_ode(x, t, m=1., b=1, k=1):
   return x[1] + mi*x[0], -x[0] + mi*x[1] - x[0]**2 * x[1]


plt.close('all')
# Generate a vector plot for the damped oscillator
plt.figure()
plt.clf()
plt.axis([-2, 2, -2, 2])
plt.title('mi=+0.1')
mi = 0.1
x0 = np.linspace (-2, 2, 20) 
x1 = np.linspace (-2, 2, 20)
Xex0 = []
for x in x0:
    for y in x1:
        Xex0.append([x, y])

#print (Xex0)
# Outer trajectories
phase_plot(
    ex3_ode,
    X0=Xex0,
    T=np.linspace(0, 100, 1000),
    logtime=(3, 0.7)
)

plt.figure()
plt.clf()
plt.axis([-2, 2, -2, 2])
plt.title('mi=-0.1')
mi = -0.1
x0 = np.linspace (-2, 2, 10) 
x1 = np.linspace (-2, 2, 10)
Xex0 = []
for x in x0:
    for y in x1:
        Xex0.append([x, y])

#print (Xex0)
# Outer trajectories
phase_plot(
    ex3_ode,
    X0=Xex0,
    T=np.linspace(0, 100, 1000),
    logtime=(3, 0.7)
)

plt.figure()
plt.clf()
plt.axis([-2, 2, -2, 2])
plt.title('mi=-0')
mi = 0
x0 = np.linspace (-1, 1, 10) 
x1 = np.linspace (-1, 1, 10)
Xex0 = []
for x in x0:
    for y in x1:
        Xex0.append([x, y])

#print (Xex0)
# Outer trajectories
phase_plot(
    ex3_ode,
    X0=Xex0,
    T=np.linspace(0, 100, 1000),
    logtime=(3, 0.7)
)

plt.figure()
plt.clf()
plt.axis([-2, 2, -2, 2])
plt.title('mi=-0.01')
mi = -0.01
x0 = np.linspace (-1, 1, 10) 
x1 = np.linspace (-1, 1, 10)
Xex0 = []
for x in x0:
    for y in x1:
        Xex0.append([x, y])

#print (Xex0)
# Outer trajectories
phase_plot(
    ex3_ode,
    X0=Xex0,
    T=np.linspace(0, 1000, 10000),
    logtime=(3, 0.7)
)

plt.show()