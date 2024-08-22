import numpy as np
import matplotlib.pyplot as plt
from control.phaseplot import phase_plot
import control as ct
from numpy import pi
import scipy.integrate as spi
def sat(x):
    if x>=1:
        return 1
    elif x<= -1:
        return -1
    return x

def ref(t):
    return 1-np.tanh(5*t)

def f(x, t):
    return 0.5*np.abs(x[0])*x[1]**2 +0.5*np.cos(2*x[0])*x[0]**3

def s(x, t, lamb):
    return x[1]+lamb*(x[0]-ref(t))

def s_(x0, x1, t, lamb):
    return s([x0, x1], t, lamb)

def u(x, t):
    K = 12
    lamb = 1
    return -f(x, t)-lamb*(x[0]-ref(t))-K*sat(s(x, t, lamb)/0.1)

def sistema(x, t, c, k):
    # c = args[0]
    # k = args[1]
    return x[1], -c*np.abs(x[0])*x[1]**2 -k*np.cos(2*x[0])*x[0]**3+u(x, t)

t = np.linspace (0, 5, 10000)
#vsis = spi.odeint (sistema, [0, 0], t, tfirst=False)
vsis = []
vsis.append([1,2])
c = -1
k = -1
for i in range (len(t)):
    dt = t[1] - t[0]
    x1_dot, x2_dot = sistema (vsis[i], t[i], c, k)
    vsis.append([vsis[i][0]+x1_dot*dt,vsis[i][1]+x2_dot*dt])

arr_out = np.transpose(np.array (vsis))
# plt.subplot(3,1,1)

plt.title("Resposta para c=-1,k=-1")
plt.xlabel("t")
plt.plot(t, arr_out[0][:-1], 'b', label='x1')
plt.plot(t, arr_out[1][:-1], 'r', label='x2')
plt.plot(t, ref(t), 'g', label='ref')
plt.grid()
plt.legend ()

# plt.subplot(3,1,2)
plt.figure()
plt.xlabel("x")
plt.ylabel("xdot")
plt.plot(arr_out[0][:-1], arr_out[1][:-1], 'r')
plt.grid()
lamb = 1
plt.figure()
# plt.subplot(3,1,3)
plt.xlabel("t")
plt.ylabel("s")
plt.plot(t, s_(arr_out[0][:-1], arr_out[1][:-1], t, lamb))

c = 1
k = 5
vsis = []
vsis.append([1,2])
for i in range (len(t)):
    dt = t[1] - t[0]
    x1_dot, x2_dot = sistema (vsis[i], t[i], c, k)
    vsis.append([vsis[i][0]+x1_dot*dt,vsis[i][1]+x2_dot*dt])

arr_out = np.transpose(np.array (vsis))
plt.figure ()
#plt.subplot(3,1,1)
plt.title("Resposta para c=1,k=5")
plt.plot(t, arr_out[0][:-1], 'b', label='x1')
plt.plot(t, arr_out[1][:-1], 'r', label='x2')
plt.plot(t, ref(t), 'g', label='ref')
plt.grid()
plt.legend ()
plt.xlabel("t")

plt.figure()

#plt.subplot(3,1,2)
plt.xlabel("x")
plt.ylabel("xdot")
plt.plot(arr_out[0][:-1], arr_out[1][:-1], 'r')
plt.grid()
lamb = 1
plt.figure()
#plt.subplot(3,1,3)
plt.xlabel("t")
plt.ylabel("s")
plt.plot(t, s_(arr_out[0][:-1], arr_out[1][:-1], t, lamb))
plt.show()