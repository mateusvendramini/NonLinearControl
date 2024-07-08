# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:51:37 2020

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
from control.phaseplot import phase_plot
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
def sat_sim1(x, t, m=1., l=1., b=0.2, g=1):
    return x[1], x[0] + x[1] - sat(2*x[0] + 2*x[1], -1, 1)

def sat_sim2(x, t, m=1., l=1., b=0.2, g=1):
    return x[1], -x[0] + 2*x[1] -sat(3*x[1], -1, 1)

def sat_sim3(x, t, m=1., l=1., b=0.2, g=1):
    return x[1], -2*x[0] - 2*x[1] -sat(-1*x[0] - 1*x[1], -1, 1)

def sat_sim4(x, t, m=1., l=1., b=0.2, g=1):
    return x[1], x[0] -sat(2*x[0] + 1*x[1], -1, 1)

def ex2_sim(x, t, m=1., l=1., b=0.2, g=1):
    return x[1] + x[0]*(1-x[0]*x[0] - 2 - x[1]*x[1]), -x[0] + x[1]*(1 - x[0]*x[0] - x[1]*x[1])

a = 1
b = 1 
def model2 (x, t):
   return x[1] + a * x[0] *(1 - 2*b - x[0]**2 - x[1]**2), -x[0] + a*x[1]*((1 - x[0]**2 - x[1]**2))

# Set up the figure the way we want it to look
# plt.figure()
# plt.clf()
# plt.axis([-2, 2, -2, 2])
# plt.title('a')
# x0 = np.linspace (-2, 2, 10) 
# x1 = np.linspace (-2, 2, 10)
# Xex0 = []
# for x in x0:
#     for y in x1:
#         Xex0.append([x, y])

# #print (Xex0)
# # Outer trajectories
# phase_plot(
#     sat_sim1,
#     X0=Xex0, #=[[-1.0, -1.0], [1, 1], [0.1, 0.1]],
#     T=np.linspace(0, 20, 100),
#     logtime=(3, 0.7)
# )


# plt.figure()
# plt.clf()
# plt.axis([-2, 2, -2, 2])
# plt.title('b')
# x0 = np.linspace (-2, 2, 10) 
# x1 = np.linspace (-2, 2, 10)
# Xex0 = []
# for x in x0:
#     for y in x1:
#         Xex0.append([x, y])

# #print (Xex0)
# # # Outer trajectories
# # phase_plot(
# #     sat_sim2,
# #     X0=Xex0, #=[[-1.0, -1.0], [1, 1], [0.1, 0.1]],
# #     T=np.linspace(0, 100, 1000),
# #     logtime=(3, 0.7)
# # )



# plt.figure()
# plt.clf()
# plt.axis([-1, 1, -1, 1])
# plt.title('c')
# x0 = np.linspace (-1, 1, 10) 
# x1 = np.linspace (-1, 1, 10)
# Xex0 = []
# for x in x0:
#     for y in x1:
#         Xex0.append([x, y])

# #print (Xex0)
# # Outer trajectories
# # phase_plot(
# #     sat_sim3,
# #     X0=Xex0, #=[[-1.0, -1.0], [1, 1], [0.1, 0.1]],
# #     T=np.linspace(0, 100, 1000),
# #     logtime=(3, 0.7)
# # )


# plt.figure()
# plt.clf()
# plt.axis([-3, 3, -3, 3])
# plt.title('d')
# x0 = np.linspace (-3, 3, 20) 
# x1 = np.linspace (-3, 3, 20)
# Xex0 = []
# for x in x0:
#     for y in x1:
#         Xex0.append([x, y])

# #print (Xex0)
# # # Outer trajectories
# # phase_plot(
# #     sat_sim4,
# #     X0=Xex0, #=[[-1.0, -1.0], [1, 1], [0.1, 0.1]],
# #     T=np.linspace(0, 100, 1000),
# #     logtime=(3, 0.7)
# # )


# plt.figure()
# plt.clf()
# plt.axis([-3, 3, -3, 3])
# plt.title('d')
# x0 = np.linspace (-3, 3, 20) 
# x1 = np.linspace (-3, 3, 20)
# Xex0 = []
# for x in x0:
#     for y in x1:
#         Xex0.append([x, y])

# #print (Xex0)
# # Outer trajectories
# phase_plot(
#     ex2_sim,
#     X0=Xex0, #=[[-1.0, -1.0], [1, 1], [0.1, 0.1]],
#     T=np.linspace(0, 100, 1000),
#     logtime=(3, 0.7)
# )


plt.figure()
plt.clf()
plt.axis([-3, 3, -3, 3])
plt.title('d')
x0 = np.linspace (-3, 3, 10) 
x1 = np.linspace (-3, 3, 10)
Xex0 = []
for x in x0:
    for y in x1:
        Xex0.append([x, y])
a=0.5
b=2
#print (Xex0)
# Outer trajectories
phase_plot(
    model2,
    X0=Xex0, #=[[-1.0, -1.0], [1, 1], [0.1, 0.1]],
    T=np.linspace(0, 100, 1000),
    logtime=(3, 0.7)
)

plt.figure()
plt.clf()
plt.axis([-3, 3, -3, 3])
plt.title('e')
x0 = np.linspace (-3, 3, 10) 
x1 = np.linspace (-3, 3, 10)
Xex0 = []
for x in x0:
    for y in x1:
        Xex0.append([x, y])
a=10
b=-10
#print (Xex0)
# Outer trajectories
phase_plot(
    model2,
    X0=Xex0, #=[[-1.0, -1.0], [1, 1], [0.1, 0.1]],
    T=np.linspace(0, 100, 1000),
    logtime=(3, 0.7)
)
# Separatrices
plt.show()
#input ()