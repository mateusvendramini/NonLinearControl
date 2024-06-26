# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 07:53:16 2020

@author: eduardo
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dy/dt
def model(y,t,u,k):
    dydt = -np.abs(y) * y + k*u
    return dydt

# initial condition
y0 = 0

# number of time points
n = 401

# time points
t = np.linspace(0,10,n)

# step input
u = np.zeros(n)
# change to 1.0 at time = 0.0
# change to 0.0 at time = 5.0
u[0:200] = 1.0

# store solution
y1 = np.empty_like(t)
y2 = np.empty_like(t)

k = 1
# solve ODE
for i in range(1,n):
    # span for next time step
    tspan = [t[i-1],t[i]]
    # solve for next step
    z = odeint(model,y0,tspan,args=(u[i],k))
    # store solution for plotting
    y1[i] = z[1][0]
    # next initial condition
    y0 = z[1]


k = 10
# solve ODE
for i in range(1,n):
    # span for next time step
    tspan = [t[i-1],t[i]]
    # solve for next step
    z = odeint(model,y0,tspan,args=(u[i],k))
    # store solution for plotting
    y2[i] = z[1][0]
    # next initial condition
    y0 = z[1]


# plot result
plt.plot(t,y1,'r-',linewidth=2,label='k=1')
plt.plot(t,y2,'b-',linewidth=2,label='k=10')
plt.xlabel('time')
plt.ylabel('y(t)')
plt.legend()
plt.show()

