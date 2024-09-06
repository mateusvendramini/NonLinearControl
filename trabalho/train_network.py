import torch
import torch.nn as nn
from torch.autograd import Variable
import torch_directml
import numpy as np
import os


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(16,20)
        self.hidden_layer2 = nn.Linear(20,18)
        self.hidden_layer3 = nn.Linear(18,16)
        self.hidden_layer4 = nn.Linear(16,14)
        self.hidden_layer5 = nn.Linear(14,12)
        self.output_layer = nn.Linear(12,8)

    def forward(self, x):
        #inputs = torch.cat([x,t],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = torch.sigmoid(self.hidden_layer1(x))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out) ## For regression, no activation is used in output layer
        return output
    
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
def H(m1, m2, L1, L2, I1, I2, F1, F2, q1, q2, q3, q4):
    g = 9.8
    a1 = I1 +(m1*L1*L1)/4 + m2*(L1**2 + (L2**2)/4)
    a2 = I2 +(m2*L2**2)/4
    a3 = m2*L1*L2/2
    
    H = np.array([[a1+2*a3*np.cos(q2), a2+a3*np.cos(q2)],
                    [a2+a3*np.cos(q2), a2]])
    return H

def f(m1, m2, L1, L2, I1, I2, F1, F2, q1, q2, q3, q4):
    g = 9.8
    a3 = m2*L1*L2/2
    a4 = g*(m1*L1/2 + m2*L1)
    a5 = m2*g*L2/2
    
    q2dd_den = F2*q4+a5*np.cos(q1+q2)+a3*np.sin(q2)*(q3**2)
    q1dd_den = F1*q3+a4*np.cos(q1)+a5*np.cos(q1+q2)+2*a3*np.sin(q2)*q3*q4+a3*np.sin(q2)*(q4*q4)

    
    return np.array([[-q1dd_den], [-q2dd_den]])



def model_loss_(x, u_hat):
    #formato de x
    #[[q1, q2, q3, q4, T1, T2, q3_nex, q4_next, parametros hat]
    

    #u_hat = net(x) # m1 m2 L1 L2 I1 I2 F1 F2 estimados
    #u_hat = u_hat.detach().numpy() 
    u_hat = u_hat.numpy() 
    x = x.numpy()
    q1 = x[0]
    q2 = x[1]
    q3 = x[2]
    q4 = x[3]
    dt = 0.000030517578125
    Hi =      H(x[8],     x[9],     x[10],    x[11],    x[12],    x[13],    x[14],    x[15],    q1, q2, q3, q4)
    Hi_next = H(u_hat[0], u_hat[1], u_hat[2], u_hat[3], u_hat[4], u_hat[5], u_hat[6], u_hat[7], q1, q2, q3, q4) #q1, q2, q3, q4
    fi_next = f(u_hat[0], u_hat[1], u_hat[2], u_hat[3], u_hat[4], u_hat[5], u_hat[6], u_hat[7], q1, q2, q3, q4) # formato 
    fi =      f(x[8],     x[9],     x[10],    x[11],    x[12],    x[13],    x[14],    x[15],    q1, q2, q3, q4)
    Hi_next_inv = np.linalg.inv(Hi_next)
    e = np.matmul(Hi_next_inv,fi_next - fi) - np.matmul(np.matmul(Hi_next_inv, Hi), np.array([[x[4]], [x[5]]])) 
    e = dt*e
    e = np.transpose(e)
    dv = np.array([[x[6]-q3, x[7]-q4]])
    #np.matmul()
    return dv, e

def model_loss(x, net, device):
    dvs = []
    es = []
    i = 0
    u =net(x)
    u = u.detach()
    x = x.detach()
    for i in range(len(x)):
        dv, e = model_loss_(x[i], u[i])
        dvs.append(dv)
        es.append(e)
        i+=1
        # if (i%10 == 0):
        #     print('step {0}'.format(i))
        #out = np.concatenate(out, np.array([[dv, e]]))
    es_ = np.array(es)
    dvs_ = np.array(dvs)
    return Variable(torch.from_numpy(dvs_).float(), requires_grad=False).to(device), Variable(torch.from_numpy(es_).float(), requires_grad=False).to(device)

# def my_mse(a, b):
#     a = a.detach().numpy()
#     b = b.detach().numpy()
#     err = a - b
#     out = 0.00
#     len_a = len(a)
#     for i in range(len_a):
#         out = np.nansum(np.array([np.dot(err[i][0], err[i][0])/len_a, out]))
#     tensor_out = torch.from_numpy(np.array([out]))
#     return tensor_out

def main():
    device = torch_directml.device() #torch.device("cpu")#torch_directml.device()
    print('using device ', device)
    #print('device name', torch_directml.device_name(0))
    set_seed(666)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    Y = np.empty((0,8))  
    X = np.empty((0,16))
    #load data
    for i in range (8):
        x_file_name='x_{0}.out.npy'.format(i)
        y_file_name='y_{0}.out.npy'.format(i)
        x_path = (os.path.join('.', 'trabalho','np_arrays2', x_file_name))
        y_path = (os.path.join('.', 'trabalho','np_arrays2', y_file_name))
        x = np.load(x_path, 'r')
        X = np.concatenate((X, x))
        y = np.load(y_path, 'r')
        Y = np.concatenate((Y, y))
        break

    print('X', len(X))
    print('Y', len(Y))
    X = X[~np.isnan(X).any(axis=1)]
    Y = Y[~np.isnan(Y).any(axis=1)]
    net = Net()
    net = net.to(device)
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(net.parameters())

    ### (3) Training / Fitting
    iterations = 100000
    previous_validation_loss = 99999999.0
    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero
        pt_x_bc = Variable(torch.from_numpy(X).float(), requires_grad=False).to(device)
        pt_y_bc = Variable(torch.from_numpy(Y).float(), requires_grad=False).to(device)
        net_bc_out = net(pt_x_bc)
        mse_u = mse_cost_function(net_bc_out, pt_y_bc)
        #e, dv = model_loss(pt_x_bc, net, device)
        #mse_f = mse_cost_function(e, dv)
        loss = mse_u #+ mse_f

        loss.backward() # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

        with torch.autograd.no_grad():
    	    print(epoch,"Traning Loss:",loss.data)


    print()
    print('saving model')
    torch.save(net.state_dict(), "model_uxt.pt")

if __name__ == '__main__':
    main()