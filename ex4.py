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
    if (x[0] == -1):
        return 0,0
    return (1-x[0])*x[1] - 2*x[0]*x[1]/(1+x[0]), x[1]*(1-x[1]/(1+x[0]))


plt.close('all')
# Generate a vector plot for the damped oscillator
plt.figure()
plt.clf()
plt.axis([0, 4, 0, 4])
plt.title('mi=+0.1')
x0 = np.linspace (0, 4, 10) 
x1 = np.linspace (0, 4, 10)
Xex0 = []
for x in x0:
    for y in x1:
        Xex0.append([x, y])

#print (Xex0)
# Outer trajectories
phase_plot(
    ex3_ode,
    X0=Xex0,
    T=np.linspace(0, 10, 100),
    logtime=(3, 0.7)
)

plt.show()