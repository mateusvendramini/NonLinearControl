import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import train_network2
#from control.phaseplot import phase_plot
#import control as ct
from numpy import pi
#import scipy.integrate as spi
from generate_training import array_folder


def sat(x):
    if x>=1:
        return 1
    elif x<= -1:
        return -1
    return x

def s(x0, x1, t, lamb, ref):
        return x1+lamb*(x0-ref(t))

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration >= total: 
        print()

class Sistema:
    def __init__(self, m1, m1h, m2, m2h, L1, L1h, L2, L2h, I1, I1h, I2, I2h, F1, F1h, F2, F2h, ref1, ref2, q10, q20, K1, K2,q30, q40, model_path):
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
        self.net = train_network2.Net()
        # internal output variables
        self._U = np.empty((0,2))  
        self._X = np.empty((0,4))
        #self.t = np.linspace (0, 1/512., 64)
        self.t = np.linspace (0, 1, 32768)
        self.m_sqerror = []
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()
        self.learning_parameters = True

        #cada vetor de saída vai ter o formato [[q1, q2, q3, q4, T1, T2, q3_nex, q4_next, parametros hat], ]

        
    def _ref1(self, t):
        return (self.ref1)*np.tanh(10*t) + self.q10

    def _ref2(self, t):
        return self.q20 + (self.ref2) * np.tanh(10*t)
    
    def update_control(self, i):
        #build input
        if (i < 2):
            return
        if (not self.learning_parameters):
            self.m_sqerror.append((self.m1h-self.m1)**2 + (self.m2h-self.m2)**2 + 
                              (self.L1h-self.L1)**2 + (self.L2h-self.L2)**2+
                              (self.I1h-self.I1)**2 + (self.I2h-self.I2)**2 + 
                              (self.F1h-self.F1)**2 + (self.F2h-self.F2)**2)
            return

        #
        x = torch.tensor([[self._X[i-1][0], self._X[i-1][1], self._X[i-1][2], self._X[i-1][3],
                self._U[i-1][0], self._U[i-1][1],
                self._X[i][2], self._X[i][3],
                (self.m1h-2)/6, (self.m2h-1)/4, self.L1h-1, self.L2h-1, (self.I1h-0.1)/0.3, (self.I2h-0.05)/0.15, (self.F1h-10)/10, (self.F2h-10)/10]]) #has to be normlz
        x = x.to(torch.float32)
        m_new = self.net(x).numpy()
        dm1 = ((m_new[0][0])*6+2 - self.m1h)
        dm2 = ((m_new[0][1])*4+1 - self.m2h)
        dL1 = ((m_new[0][2]+1) - self.L1h)
        dL2 = (m_new[0][3]+0.5 - self.L2h)
        dI1 = ((m_new[0][4])*0.3 +0.1 - self.I1h)
        dI2 = ((m_new[0][5])*0.125+0.05 - self.I2h)
        dF1 = ((m_new[0][6])*10+10 - self.F1h)
        dF2 = ((m_new[0][7])*10+10 - self.F2h)
        self.m1h = self.m1h + 0.01*dm1
        self.m2h = self.m2h + 0.01*dm2
        self.L1h = self.L1h + 0.01*dL1
        self.L2h = self.L2h + 0.01*dL2
        self.I1h = self.I1h + 0.01*dI1
        self.I2h = self.I2h + 0.01*dI2
        self.F1h = self.F1h + 0.01*dF1
        self.F2h = self.F2h + 0.01*dF2
        
        self.m_sqerror.append((self.m1h-self.m1)**2 + (self.m2h-self.m2)**2 + 
                              (self.L1h-self.L1)**2 + (self.L2h-self.L2)**2+
                              (self.I1h-self.I1)**2 + (self.I2h-self.I2)**2 + 
                              (self.F1h-self.F1)**2 + (self.F2h-self.F2)**2)
        if np.sqrt((dm1**2 + dm2**2 + dL1**2 + dL2**2 + dI1**2 + dI2**2 + dF1**2 + dF2**2)/8) < 0.00001:
            self.learning_parameters = 0
            print('finished learning parameters')
        return


    def u (self, x, t, i):
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
        
        self.update_control(i)

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

        U = q1dd_den-slin[0][0]-lamb*q1d, q2dd_den-slin[1][0]-lamb*q2d
        self._U = np.concatenate((self._U, np.array([[U[0], U[1]]])))
        return U[0], U[1]
        #return 0.8*p1-50*(x[0]-ref(t)), 0.8*p2-50*(x[1]-ref(t)) 

    def getDerivatives(self, x, t, I1, I2, L1, L2, m1, m2, F1, F2, g, i):
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
        U = self.u(x, t, i)

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
        total = len(self.t)
        printProgressBar(0, total)
        for i in range (len(self.t)):
            #print(self._X[i])
            x1_dot, x2_dot, x3_dot, x4_dot = self.getDerivatives(self._X[i], self.t[i], self.I1, self.I2, self.L1, self.L2, self.m1, self.m2, self.F1, self.F2, g, i)
            self._X= np.concatenate((self._X,
                               np.array([[self._X[i][0]+x1_dot*dt,self._X[i][1]+x2_dot*dt, self._X[i][2]+dt*x3_dot, self._X[i][3]+dt*x4_dot]])))
            if (i%40 == 0):
                printProgressBar(i, total)

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
    sis = Sistema(m1=6.0, m1h=5.0, m2=4.0, m2h=3.0, L1=1.2, L1h=1.5, L2=1, L2h=1, 
                  I1=0.25, I1h=0.25, I2=0.125, I2h=0.125, F1=15, F1h=15, 
                  F2=15, F2h=15, ref1=np.pi/6, ref2=np.pi/3, q10=0, q20=-0, K1=74, K2=266,q30=0, q40=0, model_path=os.path.join(array_folder, 'model_uxt_norm2.pt'))
    sis.run()
    #out, y = sis.getTrainingArray()
    xsis = []
    ysis = []
    v1sis = []
    v2sis = []
    #t = np.linspace (0, 1, 100000)
    #total = len(out)
    #printProgressBar(0, total)
    # for i in range(total):
    #     x1, x2, x3, x4 = out[i][0], out[i][1], out[i][2], out[i][3]
    #     xsis.append(x1)
    #     ysis.append(x2)
    #     v1sis.append(x3)
    #     v2sis.append( x4)

    #     if (i%40 == 0):
    #         printProgressBar(i, total)

    
    plt.subplot(1, 2, 1)
    plt.suptitle('Simulação controlador de modos deslizantes, grande erro parâmetros')
    plt.plot(sis.t, sis._X[:, 0][:-1], 'b', label='q1')
    plt.plot(sis.t, sis._X[:, 1][:-1], 'r', label='q2')
    plt.grid()
    plt.legend ()

    plt.subplot(1, 2, 2)
    plt.plot(sis.t, sis._X[:,2][:-1], 'b', label='q1d')
    plt.plot(sis.t, sis._X[:, 3][:-1], 'r', label='q2d')
    plt.grid()
    plt.legend ()

    plt.figure()
    plt.plot(sis.t[2:], sis.m_sqerror, 'r', label='m_squared')
    plt.show()

    print("parametros estimados no final {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}".format(
        sis.m1h, sis.m2h, sis.L1h, sis.L2h, sis.I1h, sis.I2h, sis.F1h, sis.F2h
    ))

if __name__ == '__main__':
    main()