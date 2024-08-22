import os

import numpy as np
import matplotlib.pyplot as plt
from control.phaseplot import phase_plot
import control as ct
from numpy import pi
import scipy.integrate as spi
def feedback_linearization(x,t):
    return -2*(x[0]**2 + x[1])*x[0]

def ref(t):
    return np.tanh(30*t)

# def u(x, t):
#     global last_x
#     global last_t
#     if last_t == t:
#         last_x = x[0]
#         last_t = t
#         #print ("last x " + last_x + "last_t" + last_t)
#         return feedback_linearization(x, t) + ref(t) - x[0]  
    
#     sig = feedback_linearization(x, t)  + 2* (last_x-x[0])/(t-last_t) + ref(t) - x[0]
#     last_x = x[0]
#     last_t = t
#     return sig 
def u (x, t, y_dot):
    # global last_t
    # global last_x
    # if last_t == t:
    #     last_x = x[0]
    #     last_t = t
    #     return ref (t) - x[0] + feedback_linearization(x, t)
    # derivative = (x[0]-last_x)/(t - last_t)
    # print ("\r\nderivative %f time %f" %(derivative, t))
    # last_x = x[0]
    # last_t = t
    return  10*(ref (t) - x[0]) + feedback_linearization(x, t) - 11*y_dot

last_x1_dot = 0.00
def sistema(x, t):
    global last_x1_dot 
    last_x1_dot_old = last_x1_dot

    last_x1_dot = x[0] ** 2 + x[1]
    #print ("\r\n %f x1d %f x2d %f" %(t, last_x1_dot, u(x, t, last_x1_dot_old)))
    return last_x1_dot, u(x, t, last_x1_dot_old)

last_x = 0.00
last_t = 0.00
# print (str(u([0, 0], 0)))
# print (str(u([0, 0], 0.1)))

t = np.linspace (0, 5, 500)
#vsis = spi.odeint (sistema, [0, 0], t, tfirst=False)
vsis = []
vsis.append([0,0])

for i in range (len(t)):
    dt = t[1] - t[0]
    x1_dot, x2_dot = sistema (vsis[i], t[i])
    vsis.append([vsis[i][0]+x1_dot*dt,vsis[i][1]+x2_dot*dt])

arr_out = np.transpose(np.array (vsis))

plt.plot (t, ref(t), 'g', label='ref')
plt.plot(t, arr_out[0][:-1], 'b', label='x1')
plt.plot(t, arr_out[1][:-1], 'r', label='x2')
plt.grid()
plt.legend ()

plt.show()