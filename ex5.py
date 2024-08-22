import os

import numpy as np
import matplotlib.pyplot as plt
from control.phaseplot import phase_plot
import control as ct
from numpy import pi
import scipy.integrate as spi



# def oscillator_ode(x, t, m=1., b=1, k=1):
#     return x[1], -k/m*x[0] - b/m*x[1]
M = 5
C = 400
K = 2000

def ex5_ode(x, t, m=1., b=1, k=1):
    return x[1], -400*x[1]/M - (K/M) * x[0] ** 3 - (C/M)*(x[0]**2 -1)*x[1]

def ex5_ode_ct2 (x, t, m=1., b=1, k=1):
    return x[1], -600*x[1]/M -1000*x[0]/M - (K/M) * x[0] ** 3 - (C/M)*(x[0]**2 -1)*x[1]

plt.figure()
plt.clf()
plt.axis([0, 10, 0, 10])
plt.title('Resposta para condição inicial (10, 1) K=1000 C=200')

t = np.linspace (0, 10, 100)
x0 = 10
y0 = 1
K = 1000
C = 200

v = spi.odeint (ex5_ode, [x0, y0], t, tfirst=False)
plt.plot(t, v[:, 0], 'b', label='x1')
# plt.plot(t, v[:, 1], 'r', label='x2')


plt.figure()
plt.clf()
plt.axis([0, 10, 0, 10])
plt.title('Resposta para condição inicial (10, 1) K=2000 C=200')
K = 2000
C = 200

t = np.linspace (0, 10, 100)
x0 = 10
y0 = 1
v = spi.odeint (ex5_ode, [x0, y0], t, tfirst=False)
plt.plot(t, v[:, 0], 'b', label='x1')
# plt.plot(t, v[:, 1], 'r', label='x2')


plt.figure()
plt.clf()
plt.axis([0, 10, -1, 10])
plt.title('Resposta para condição inicial (10, 1) K=1000 C=400')
K = 1000
C = 400

t = np.linspace (0, 10, 100)
x0 = 10
y0 = 1
v = spi.odeint (ex5_ode, [x0, y0], t, tfirst=False)
plt.plot(t, v[:, 0], 'b', label='x1')
# plt.plot(t, v[:, 1], 'r', label='x2')



plt.figure()
plt.clf()
plt.axis([0, 10, -1, 10])
plt.title('Resposta para condição inicial (10, 1) K=2000 C=400')
K = 2000
C = 400

t = np.linspace (0, 10, 100)
x0 = 10
y0 = 1
v = spi.odeint (ex5_ode, [x0, y0], t, tfirst=False)
plt.plot(t, v[:, 0], 'b', label='x1')


plt.figure()
plt.clf()
plt.axis([0, 10, -2, 10])
plt.title('Resposta para condição inicial (10, 1) K=1000 C=200 controle 2')
K = 1000
C = 200

t = np.linspace (0, 10, 100)
x0 = 10
y0 = 1
v = spi.odeint (ex5_ode_ct2, [x0, y0], t, tfirst=False)
plt.plot(t, v[:, 0], 'b', label='x1')
#plt.plot(t, v[:, 1], 'r', label='x2')

plt.show()