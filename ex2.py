# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:08:03 2020

@author: eduardo
"""


# phaseplots.py - examples of phase portraits
# RMM, 24 July 2011
#
# This file contains examples of phase portraits pulled from "Feedback
# Systems" by Astrom and Murray (Princeton University Press, 2008).

import os

import numpy as np
import matplotlib.pyplot as plt
from control.phaseplot import phase_plane_plot
import control as ct
from numpy import pi

# Clear out any figures that are present
plt.close('all')

#
# Inverted pendulum
#
def sat (y, min, max):
   if y < min:
      return min
   if y > max:
      return max
   return y

# Define the ODEs for a damped (inverted) pendulum
def model(t, x, u, params):
# return x[1], -2*c*(x[0]**2-1)*x[1] - k*x[0]
# return x[1], -0.6*x[1] -3*x[0] -x[0]**2
#    return x[1], -0.1*x[1] - x[0]
 return np.array(x[1], x[0] + x[1] - sat(2*x[0] + 2*x[1], 0, 1))
a = 1
b = 1 
def model2 (t, x, u, params):
   return np.array(x[1] + a * x[0] *(1 - 2*b - x[0]**2 - x[1]**2), -x[0] + a*x[1]*((1 - x[0]**2 - x[1]**2))) 

ex1 = ct.nlsys(model, states=['position', 'velocity'], inputs=0)
ex2 = ct.nlsys(model2, states=['x', 'y'], inputs=0)

# Set up the figure the way we want it to look
plt.figure()
plt.clf()
plt.axis([-3, 3, -3, 3])
plt.title('a')
plt.axis('Equal')

# Outer trajectories

x0vec = np.linspace(-3, 3, 100)
y0vec = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x0vec, y0vec)
x0 = []
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x0.append((X[i,j], Y[i,j]))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#fig.set_tight_layout(True)
plt.suptitle("FBS Figure 5.3: a=1,b=1")


phase_plane_plot(
    ex1,
     [-3, 3, -3, 3], 100, gridtype='meshgrid'
)
ax1.set_title("boxgrid [-3, 3, -3, 3], 8")

#ct.phase_plane_plot(ex1, [-1, 1, -1, 1], ax=ax2, gridtype='meshgrid')
#ax2.set_title("meshgrid [-1, 1, -1, 1]")

# ct.phase_plane_plot(
#     ex1, [-1, 1, -1, 1], 4, ax=ax3, gridtype='circlegrid', dir='both')
# ax3.set_title("circlegrid [0, 0, 1], 4, both")

# ct.phase_plane_plot(
#     ex1, [-1, 1, -1, 1], ax=ax4, gridtype='circlegrid',
#     dir='reverse', gridspec=[0.1, 12], timedata=5)
# ax4.set_title("circlegrid [0, 0, 0.1], reverse")

