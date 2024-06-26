#teste massa mola 
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import control as ct
from control.matlab import *    # MATLAB-like functions

#from matlab import engine

# G(s) = 1/(s^2 + 4s + 1) em malha fechada

num = [1.0]
den = [1.0, 4.0, 1.0]
sys1 = tf(num, den)
yout, T = impulse(sys1, T=100)
plt.plot(T, yout,'r-',linewidth=2,label='r=1')
#plt.plot(t,y2,'b-',linewidth=2,label='k=10')
plt.xlabel('time')
plt.ylabel('y(t)')
plt.legend()
plt.show()
roots, gains = rlocus (sys1)
plt.plot (roots, gains)
plt.show()