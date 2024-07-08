import numpy as np
import matplotlib.pyplot as plt
#a = -1
#b = -10
plt.close('all')

def dF(r, theta, a, b):
    return a*r*(1-r*r - 2*b*np.cos(theta)), a*b*np.sin(2*theta) - 1


X, Y = np.meshgrid(np.linspace(-3.0, 3.0, 30), np.linspace(-3.00, 3.0, 30))
R, Theta = (X**2 + Y**2)**0.5, np.arctan2(Y, X)
dR, dTheta = dF(R, Theta, 1, 1)
C, S = np.cos(Theta), np.sin(Theta)
U, V = dR*C - R*S*dTheta, dR*S+R*C*dTheta
plt.figure()
plt.streamplot(X, Y, U, V, color='r', linewidth=0.5, density=1.6)
plt.axis('square')
plt.axis([-3, 3, -3, 3])
plt.title('a=1,b=1')

X, Y = np.meshgrid(np.linspace(-3.0, 3.0, 30), np.linspace(-3.00, 3.0, 30))
R, Theta = (X**2 + Y**2)**0.5, np.arctan2(Y, X)
dR, dTheta = dF(R, Theta, 0.5, 2)
C, S = np.cos(Theta), np.sin(Theta)
U, V = dR*C - R*S*dTheta, dR*S+R*C*dTheta
plt.figure()
plt.streamplot(X, Y, U, V, color='r', linewidth=0.5, density=1.6)
plt.axis('square')
plt.axis([-3, 3, -3, 3])
plt.title('a=0.5,b=2')


X, Y = np.meshgrid(np.linspace(-3.0, 3.0, 30), np.linspace(-3.00, 3.0, 30))
R, Theta = (X**2 + Y**2)**0.5, np.arctan2(Y, X)
dR, dTheta = dF(R, Theta, -1, 2)
C, S = np.cos(Theta), np.sin(Theta)
U, V = dR*C - R*S*dTheta, dR*S+R*C*dTheta
plt.figure()
plt.streamplot(X, Y, U, V, color='r', linewidth=0.5, density=1.6)
plt.axis('square')
plt.axis([-3, 3, -3, 3])
plt.title('a=-1,b=2')

X, Y = np.meshgrid(np.linspace(-3.0, 3.0, 30), np.linspace(-3.00, 3.0, 30))
R, Theta = (X**2 + Y**2)**0.5, np.arctan2(Y, X)
dR, dTheta = dF(R, Theta, (0.5)**3, -8)
C, S = np.cos(Theta), np.sin(Theta)
U, V = dR*C - R*S*dTheta, dR*S+R*C*dTheta
plt.figure()
plt.streamplot(X, Y, U, V, color='r', linewidth=0.5, density=1.6)
plt.axis('square')
plt.axis([-3, 3, -3, 3])
plt.title('a=0.125,b=-8')

# X, Y = np.meshgrid(np.linspace(-3.0, 3.0, 30), np.linspace(-3.00, 3.0, 30))
# R, Theta = (X**2 + Y**2)**0.5, np.arctan2(Y, X)
# dR, dTheta = dF(R, Theta, 1, 10)
# C, S = np.cos(Theta), np.sin(Theta)
# U, V = dR*C - R*S*dTheta, dR*S+R*C*dTheta
# plt.figure()
# plt.streamplot(X, Y, U, V, color='r', linewidth=0.5, density=1.6)
# plt.axis('square')
# plt.axis([-3, 3, -3, 3])
# plt.title('a=1,b=10')


# X, Y = np.meshgrid(np.linspace(-3.0, 3.0, 30), np.linspace(-3.00, 3.0, 30))
# R, Theta = (X**2 + Y**2)**0.5, np.arctan2(Y, X)
# dR, dTheta = dF(R, Theta, 10, 10)
# C, S = np.cos(Theta), np.sin(Theta)
# U, V = dR*C - R*S*dTheta, dR*S+R*C*dTheta
# plt.figure()
# plt.streamplot(X, Y, U, V, color='r', linewidth=0.5, density=1.6)
# plt.axis('square')
# plt.axis([-3, 3, -3, 3])
# plt.title('a=10,b=10')

plt.show()