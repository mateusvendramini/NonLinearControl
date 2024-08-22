import os

import numpy as np
import matplotlib.pyplot as plt
from control.phaseplot import phase_plot
import control as ct
from numpy import pi
import scipy.integrate as spi

def v (x, ref):
    return 10.8*np.sin (x)/0.75 - (((0.039/2)**2)/0.75)*(x-ref)

def v2 (x, v, ref):
    return 10.8*np.sin (x)/0.75 - (((0.039)**2)/0.75)*(x-ref) - 0.039*v/0.75


def vmax (x, v, ref):
    return 10.8*np.sin (x)/0.75 - ((34.3)/0.75)*(x-ref) - 11.7*v/0.75

def sistema (x, t):
    return x[1], 0.75*v(x[0], pi*np.tanh(t*10)/6) -10.8*np.sin(x[0]) - 0.039*x[1]

def sistema2 (x, t):
    return x[1], 0.75*v2(x[0], x[1], pi*np.tanh(t*10)/6) -10.8*np.sin(x[0]) - 0.039*x[1]

def sistemamax (x, t):
    return x[1], 0.75*vmax(x[0], x[1], pi*np.tanh(t*100)/6) -10.8*np.sin(x[0]) - 0.039*x[1]


plt.figure()
plt.clf()
#plt.axis([0, 10, 0, 10])
plt.title('Resposta do sistema com amortecimento')

t = np.linspace (0, 1000, 1000)
vsis = spi.odeint (sistema, [0, 0], t, tfirst=False)
plt.plot(t, vsis[:,0], 'b', label='x1')
vsis2 = spi.odeint (sistema2, [0, 0], t, tfirst=False)
plt.plot(t, vsis2[:,0], 'r', label='sistema 2')
plt.legend ()

plt.figure()
plt.clf()
#plt.axis([0, 10, 0, 10])
plt.title('Esforço do controlador com amortecimento')
plt.plot(t, v(vsis[:,0], pi*np.tanh(t*10)/6), 'b', label='Controlador Proporcional')
plt.plot(t, v2(vsis2[:,0],vsis2[:,1], pi*np.tanh(t*10)/6), 'r', label='controlador PD')
plt.legend ()

plt.figure()
plt.clf()
#plt.axis([0, 10, 0, 10])
t = np.linspace (0, 2, 1000)
vsismax = spi.odeint (sistemamax, [0, 0], t, tfirst=False)
plt.title('Resposta do sistema')
plt.plot(t, vsismax[:,0], 'r', label='sistema max')

plt.figure()
plt.title('Esforço do controlador com kmax')
plt.plot(t, vmax(vsismax[:,0], vsismax[:,1], pi*np.tanh(t*100)/6), 'b', label='Controlador Proporcional')


plt.show()