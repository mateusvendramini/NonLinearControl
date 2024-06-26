import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def motor(t, t0, x, x0):
    return (t-t0)* x + x0

def valve(t, t0, x, x0):
    return x*x
def process(t, t0, x, x0):
    return (x*np.exp(-x))

#r = 1 #step unitario para t >= 0
n = 10000
t = np.linspace (0, 5/10000000, n)

# step input
r = np.zeros(n)
# change to 1.0 at time = 0.0
# change to 0.0 at time = 10
r[0:n] = 1.0

y0 = 0
y = np.zeros(n+1)
#print (y)
#print ("y1" + str(y[1]))
errors = np.zeros(n)
motor_outputs = np.zeros(n)
valve_outputs = np.zeros(n)
last_error = 0
last_motor_output = 0
for i in range (1,n):
    #print ("step %d\r\n" %i)
    errors[i] = r[i-1] - y[i-1]
    motor_outputs[i] = motor (t[i], t[i-1], errors[i], errors[i-1])
    valve_outputs[i] = valve (t[i], t[i-1], motor_outputs[i], motor_outputs[i-1])
    y[i] = process (t[i], t[i-1], valve_outputs[i-1], y[i-1])
    last_motor_output = motor_outputs[i]
    last_error = errors[i]

# plot result
plt.plot(t,y[0:n],'r-',linewidth=2,label='r=1')
#plt.plot(t,y2,'b-',linewidth=2,label='k=10')
plt.xlabel('time')
plt.ylabel('y(t)')
plt.legend()
plt.show()

plt.plot(t,errors,'r-',linewidth=2,label='r=1')
#plt.plot(t,y2,'b-',linewidth=2,label='k=10')
plt.xlabel('time')
plt.ylabel('error(t)')
plt.legend()
plt.show()


