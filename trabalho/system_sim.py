#import os

import numpy as np
import matplotlib.pyplot as plt
#from control.phaseplot import phase_plot
#import control as ct
from numpy import pi
#import scipy.integrate as spi

def sat(x):
    if x>=1:
        return 1
    elif x<= -1:
        return -1
    return x

def s(x0, x1, t, lamb, ref):
        return x1+lamb*(x0-ref(t))

class Sistema:
    def __init__(self, m1, m1h, m2, m2h, L1, L1h, L2, L2h, I1, I1h, I2, I2h, F1, F1h, F2, F2h, ref1, ref2, q10, q20, K1, K2,q30, q40):
        '''
        m1, m1h, m2, m2h, L1, L1h, L2, L2h, I1, I1h, I2, I2h, F1, F1h, F2, F2h, ref1, ref2, q10, q20, K1, K2
        '''
        self.m1 = m1
        self.m2 = m2
        self.m1h = m1h
        self.m2h = m2h
        self.L1 = L1
        self.L1h = L1h
        self.L2 = L2
        self.L2h = L2h
        self.I1 = I1
        self.I1h = I1h
        self.I2 = I2
        self.I2h = I2h
        self.F1 = F1
        self.F1h = F1h
        self.F2 = F2
        self.F2h = F2h
        self.ref1 = ref1
        self.ref2 = ref2
        self.q10 = q10
        self.q20 = q20
        self.q30 = q30
        self.q40 = q40
        self.K1 = K1
        self.K2 = K2
        # internal output variables
        self._U = np.empty((0,2))  
        self._X = np.empty((0,4))
        self.t = np.linspace (0, 1/512., 64)

        #cada vetor de saída vai ter o formato [[q1, q2, q3, q4, T1, T2, q3_nex, q4_next, parametros hat], ]

        
    def _ref1(self, t):
        return (self.ref1)*np.tanh(10*t) + self.q10

    def _ref2(self, t):
        return self.q20 + (self.ref2) * np.tanh(10*t)

    def u (self, x, t):
        #return 0,0
        m1 = self.m1h
        m2 = self.m2h
        L1 = self.L1h
        L2 = self.L2h
        I1 = self.I1h
        I2 = self.I2h
        F1= self.F1h
        F2= self.F2h
        g = 9.8

        a1 = I1 +(m1*L1*L1)/4 + m2*(L1**2 + (L2**2)/4)
        a2 = I2 +(m2*L2**2)/4
        a3 = m2*L1*L2/2
        a4 = g*(m1*L1/2 + m2*L1)
        a5 = m2*g*L2/2

        q1 = x[0]
        q2 = x[1]
        q1d = x[2]
        q2d = x[3]
        lamb = 8
        phi = 0.01

        
        q2dd_den = F2*q2d+a5*np.cos(q1+q2)+a3*np.sin(q2)*(q1d**2)
        q1dd_den = F1*q1d+a4*np.cos(q1)+a5*np.cos(q1+q2)+2*a3*np.sin(q2)*q1d*q2d+a3*np.sin(q2)*(q2d*q2d)

        H = np.array([[a1+2*a3*np.cos(q2), a2+a3*np.cos(q2)],
                    [a2+a3*np.cos(q2), a2]])
        sden = np.array([[self.K1*sat(s(q1, q1d, t, lamb, self._ref1)/phi)+lamb*q1d], [self.K2*sat(s(q2, q2d, t, lamb, self._ref2)/phi)+lamb*q2d]])

        slin = np.matmul(H, sden)
        #s = np.array([[s(q1, q1d, t, lamb)], [s(q2, q2d, t, lamb)]])

        U = q1dd_den-slin[0][0], q2dd_den-slin[1][0]-lamb*q2d
        self._U = np.concatenate((self._U, np.array([[U[0], U[1]]])))
        return U[0], U[1]
        #return 0.8*p1-50*(x[0]-ref(t)), 0.8*p2-50*(x[1]-ref(t)) 

    def getDerivatives(self, x, t, I1, I2, L1, L2, m1, m2, F1, F2, g):
        '''Estados q1, q2, q1dot, q2dot
        '''
        a1 = I1 +(m1*L1*L1)/4 + m2*(L1**2 + (L2**2)/4)
        a2 = I2 +(m2*L2**2)/4
        a3 = m2*L1*L2/2
        a4 = g*(m1*L1/2 + m2*L1)
        a5 = m2*g*L2/2
        q1=x[0]
        q2=x[1]
        q1d=x[2]
        q2d=x[3]
        U = self.u(x, t)

        q2dd_den = U[1]-F2*q2d-a5*np.cos(q1+q2)-a3*np.sin(q2)*(q1d**2)
        q1dd_den = U[0]-F1*q1d-a4*np.cos(q1)-a5*np.cos(q1+q2)+2*a3*np.sin(q2)*q1d*q2d+a3*np.sin(q2)*(q2d*q2d)
        H = np.array([[a1+2*a3*np.cos(q2), a2+a3*np.cos(q2)],
                    [a2+a3*np.cos(q2), a2]])
        H_inv = np.linalg.inv(H)
        qdd = np.matmul(H_inv, np.array([[q1dd_den], [q2dd_den]]))
        return q1d, q2d, qdd[0][0], qdd[1][0]

    # print (str(u([0, 0], 0)))
    # print (str(u([0, 0], 0.1)))

    def run(self):
        
        self._X = np.concatenate((self._X , np.array([[self.q10, self.q20, self.q30, self.q40]])))

        g = 9.8
        dt = self.t[1] - self.t[0]

        for i in range (len(self.t)):
            #print(self._X[i])
            x1_dot, x2_dot, x3_dot, x4_dot = self.getDerivatives(self._X[i], self.t[i], self.I1, self.I2, self.L1, self.L2, self.m1, self.m2, self.F1, self.F2, g)
            self._X= np.concatenate((self._X,
                               np.array([[self._X[i][0]+x1_dot*dt,self._X[i][1]+x2_dot*dt, self._X[i][2]+dt*x3_dot, self._X[i][3]+dt*x4_dot]])))

    def getTrainingArray(self):
        out = np.empty((0,16))
        Y = np.empty((0, 8))
        y = np.array([[self.m1, self.m2, self.L1, self.L2, self.I1, self.I2, self.F1, self.F2]])

        # Cada elemento de saída vai ter o formato [[q1, q2, q3, q4, T1, T2, q3_nex, q4_next, parametros hat], ]
        for i in range(len(self._X)-1):
            out = np.concatenate((out, np.array([[
                self._X[i][0], self._X[i][1], self._X[i][2], self._X[i][3],
                self._U[i][0], self._U[i][1],
                self._X[i+1][2], self._X[i+1][3],
                self.m1h, self.m2h, self.L1h, self.L2h, self.I1h, self.I2h, self.F1h, self.F2h
            ]])))
            
            Y = np.concatenate((Y, y))
        
        return out, Y

def main():
#    m1 = 5
#m2 = 3
#L1 = 1.5
#L2 = 1
#I1 = 0.25
#I2 = 0.125
#F1= 15
#F2=15

#pyt
    # Inicia sistema
    #2.0, 2.0, 1.0, 1.0,L1 1.0, L1 1.0, L2 0.5, L2 0.5, I1 0.1, I1 0.1, i2 0.05, i2 0.05, F1 10.0, F1 10.0, F2 10.0, F2 10.0, -1.0471975511965976, -1.5707963267948966, 0.0, 0.0, 74.0, 266

    # sis = Sistema(m1=5, m1h=5, m2=3, m2h=3, L1=1.5, L1h=1.5, 
    #               L2=1, L2h=1, I1=0.25, I1h=0.25, I2=0.125, I2h=0.125, F1=15, F1h=15, F2=15 , F2h=15, ref1=np.pi/2, ref2=-np.pi/2,
    #                 q10=np.pi/2, q20=np.pi/2, K1=62, K2=254)
    sis = Sistema(m1=2.0975561637446507, m1h=1.9178533132322388, m2=0.9863902114286804, m2h=0.9651596624103864, L1=0.9165574998344682, L1h=1.0233902423594152, L2=0.49447895383476115, L2h=0.4404871475561963, I1=0.1033831643081498, I1h=0.1036096982291639, I2=0.04012892953284832, I2h=0.05330792550109102, F1=10, F1h=10, F2=10, F2h=10, ref1=np.pi/3, ref2=-np.pi/2, q10=1.127572188422499, q20=-0.02717109077810674, K1=74, K2=266)
    sis.run()
    out, y = sis.getTrainingArray()
    xsis = []
    ysis = []
    v1sis = []
    v2sis = []
    #t = np.linspace (0, 1, 100000)
    for i in range(len(out)):
        x1, x2, x3, x4 = out[i][0], out[i][1], out[i][2], out[i][3]
        xsis.append(x1)
        ysis.append(x2)
        v1sis.append(x3)
        v2sis.append( x4)
    
    plt.subplot(1, 2, 1)
    plt.suptitle('Simulação controlador de modos deslizantes, grande erro parâmetros')
    plt.plot(sis.t, xsis, 'b', label='q1')
    plt.plot(sis.t, ysis, 'r', label='q2')
    plt.grid()
    plt.legend ()

    plt.subplot(1, 2, 2)
    plt.plot(sis.t, v1sis, 'b', label='q1d')
    plt.plot(sis.t, v2sis, 'r', label='q2d')
    plt.grid()
    plt.legend ()

    plt.show()

if __name__ == '__main__':
    main()