import time
import warnings
from math import pi, sqrt

import matplotlib.pyplot as plt
import numpy as np

import control as ct
import control.phaseplot as pp

#
# Example 1: Dampled oscillator systems
#

# Oscillator parameters
damposc_params = {'m': 1, 'b': 1, 'k': 1}
def sat (y, min=0, max=1):
   if y < min:
      return min
   if y > max:
      return max
   return y

# System model (as ODE)
def damposc_update(t, x, u, params):
    m, b, k = params['m'], params['b'], params['k']
    return np.array(x[1], x[0] + x[1] - sat(2*x[0] + 2*x[1]))
damposc = ct.nlsys(damposc_update, states=2, inputs=0, params=damposc_params)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.set_tight_layout(True)
plt.suptitle("FBS Figure 5.3: damped oscillator")

ct.phase_plane_plot(damposc, [-1, 1, -1, 1], 8, ax=ax1)
ax1.set_title("boxgrid [-1, 1, -1, 1], 8")

ct.phase_plane_plot(damposc, [-1, 1, -1, 1], ax=ax2, gridtype='meshgrid')
ax2.set_title("meshgrid [-1, 1, -1, 1]")

ct.phase_plane_plot(
    damposc, [-1, 1, -1, 1], 4, ax=ax3, gridtype='circlegrid', dir='both')
ax3.set_title("circlegrid [0, 0, 1], 4, both")

ct.phase_plane_plot(
    damposc, [-1, 1, -1, 1], ax=ax4, gridtype='circlegrid',
    dir='reverse', gridspec=[0.1, 12], timedata=5)
ax4.set_title("circlegrid [0, 0, 0.1], reverse")

plt.show(block=False)
input()