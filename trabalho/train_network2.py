import torch
import torch.nn as nn
from torch.autograd import Variable
import torch_directml
import numpy as np
import os

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, low_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.low_dloss_counter = 0
        self.low_delta = low_delta
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            if (self.min_validation_loss - validation_loss < self.low_delta):
                self.low_dloss_counter+=1
                if self.low_dloss_counter >= self.patience:
                    return True
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

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


def H_(m1, m2, L1, L2, I1, I2, F1, F2, q1, q2, q3, q4):
    g = 9.8
    a1 = I1 +(m1*L1*L1)/4 + m2*(L1*L1 + (L2*L2)/4)
    a2 = I2 +(m2*L2**2)/4
    a3 = m2*L1*L2/2
    
    #H = np.array([[a1+2*a3*np.cos(q2), a2+a3*np.cos(q2)],
    #                [a2+a3*np.cos(q2), a2]])
    H = a1+2*a3*torch.cos(q2), a2+a3*torch.cos(q2), a2+a3*torch.cos(q2), a2 #return H11, H12, H21, H22
    return H

# def H(u, x):
#     #return H(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8])
#     return H_(torch.index_select(u, 1, 0), torch.index_select(u, 1, 1), 
#              torch.index_select(u, 1, 2), torch.index_select(u, 1, 3),
#              torch.index_select(u, 1, 4), torch.index_select(u, 1, 5), 
#              torch.index_select(u, 1, 6), torch.index_select(u, 1, 7),
#              torch.index_select(x, 1, 0), torch.index_select(x, 1, 1), torch.index_select(x, 1, 2), torch.index_select(x, 1, 3)
#              )

def f_(m1, m2, L1, L2, I1, I2, F1, F2, q1, q2, q3, q4):
    g = 9.8
    a3 = m2*L1*L2/2
    a4 = g*(m1*L1/2 + m2*L1)
    a5 = m2*g*L2/2
    
    q2dd_den = F2*q4+a5*torch.cos(q1+q2)+a3*torch.sin(q2)*(q3*q3)
    q1dd_den = F1*q3+a4*torch.cos(q1)+a5*torch.cos(q1+q2)+2*a3*torch.sin(q2)*q3*q4+a3*torch.sin(q2)*(q4*q4)

    
    return (-q1dd_den, -q2dd_den)

# def f(u, x):
#     return f_(torch.index_select(u, 1, 0), torch.index_select(u, 1, 1), 
#              torch.index_select(u, 1, 2), torch.index_select(u, 1, 3),
#              torch.index_select(u, 1, 4), torch.index_select(u, 1, 5), 
#              torch.index_select(u, 1, 6), torch.index_select(u, 1, 7),
#              torch.index_select(x, 1, 0), torch.index_select(x, 1, 1), torch.index_select(x, 1, 2), torch.index_select(x, 1, 3))

def model_loss_(x, u_hat, device):
    #formato de x
    #[[q1, q2, q3, q4, T1, T2, q3_nex, q4_next, parametros hat]
    

    #u_hat = net(x) # m1 m2 L1 L2 I1 I2 F1 F2 estimados
    #u_hat = u_hat.detach().numpy() 
    #u_hat = u_hat.numpy() 
    #x = x.numpy()
    x = x.cpu()
    u_hat = u_hat.cpu()

    q1 = torch.index_select(x, 1, torch.tensor([0])).to(device)
    q2 = torch.index_select(x, 1, torch.tensor([1])).to(device)
    q3 = torch.index_select(x, 1, torch.tensor([2])).to(device)
    q4 = torch.index_select(x, 1, torch.tensor([3])).to(device)
    T1 = torch.index_select(x, 1, torch.tensor([4])).to(device)
    T2 = torch.index_select(x, 1, torch.tensor([5])).to(device)
    v1 = torch.index_select(x, 1, torch.tensor([6])).to(device)
    v2 = torch.index_select(x, 1, torch.tensor([7])).to(device)

    m1x = torch.index_select(x, 1, torch.tensor([8])).to(device)
    m2x = torch.index_select(x, 1, torch.tensor([9])).to(device)
    L1x = torch.index_select(x, 1, torch.tensor([10])).to(device)
    L2x = torch.index_select(x, 1, torch.tensor([11])).to(device)
    I1x = torch.index_select(x, 1, torch.tensor([12])).to(device)
    I2x = torch.index_select(x, 1, torch.tensor([13])).to(device)
    F1x = torch.index_select(x, 1, torch.tensor([14])).to(device)
    F2x = torch.index_select(x, 1, torch.tensor([15])).to(device)
    
    m1u = torch.index_select(u_hat, 1, torch.tensor([0])).to(device)
    m2u = torch.index_select(u_hat, 1, torch.tensor([1])).to(device)
    L1u = torch.index_select(u_hat, 1, torch.tensor([2])).to(device)
    L2u = torch.index_select(u_hat, 1, torch.tensor([3])).to(device)
    I1u = torch.index_select(u_hat, 1, torch.tensor([4])).to(device)
    I2u = torch.index_select(u_hat, 1, torch.tensor([5])).to(device)
    F1u = torch.index_select(u_hat, 1, torch.tensor([6])).to(device)
    F2u = torch.index_select(u_hat, 1, torch.tensor([7])).to(device)

    
    # u_from_x = torch.tensor([
    #     torch.index_select(x, 1, torch.tensor([8])), torch.index_select(x, 1, torch.tensor([9])),
    #     torch.index_select(x, 1, torch.tensor([10])), torch.index_select(x, 1, torch.tensor([11])),
    #     torch.index_select(x, 1, torch.tensor([12])), torch.index_select(x, 1, torch.tensor([13])),
    #     torch.index_select(x, 1, torch.tensor([14])), torch.index_select(x, 1, torch.tensor([15]))
    # ])
    #q1 = x[0]
    #q2 = x[1]
    #q3 = x[2]
    #q4 = x[3]
    dt = 0.000030517578125
    Hi11, Hi12, Hi21, Hi22 = H_(m1u, m2u, L1u, L2u, I1u, I2u, F1u, F2u, q1, q2, q3, q4)
    Hi_next11, Hi_next12, Hi_next21, Hi_next22 = H_(m1x, m2x, L1x, L2x, I1x, I2x, F1x, F2x, q1, q2, q3, q4)#.to(device) #q1, q2, q3, q4
    fi_next1, fi_next2 = f_(m1x, m2x, L1x, L2x, I1x, I2x, F1x, F2x, q1, q2, q3, q4) # formato 
    fi1, fi2  =      f_(m1u, m2u, L1u, L2u, I1u, I2u, F1u, F2u, q1, q2, q3, q4)#.to(device)

    #Hi = Hi.to(device)
    den = 1/(Hi_next11*Hi_next22 - Hi_next12*Hi_next21)
    Hi_next_inv11 = den * Hi_next22
    Hi_next_inv12 = -den * Hi_next12
    Hi_next_inv21 =-den * Hi_next21
    Hi_next_inv22 = den * Hi_next11

    #e = torch.matmul(Hi_next_inv,fi_next - fi) - torch.matmul(torch.matmul(Hi_next_inv, Hi), torch.tensor([[torch.index_select(x, 1, 4)], [torch.index_select(x, 1, 5)]])) 
    HH11 = Hi_next_inv11*Hi11 + Hi_next_inv12*Hi21
    HH12 = Hi_next_inv11*Hi12 + Hi_next_inv12*Hi22
    HH21 = Hi_next_inv21*Hi11 + Hi_next_inv22*Hi21
    HH22 =  Hi_next_inv21*Hi12 + Hi_next_inv22*Hi22

    e1 = (Hi_next_inv11*(fi1-fi_next1) + Hi_next_inv12*(fi2-fi_next2) - (HH11*T1 + HH12*T2))*dt
    e2 = (Hi_next_inv21*(fi1-fi_next1) + Hi_next_inv22*(fi2-fi_next2) - (HH21*T1 + HH22*T2))*dt
    dv1 = v1 - q3
    dv2 = v2 - q4
    #e = dt*e
    #e = torch.transpose(e)
    #dv = torch.tensor([[torch.index_select(x, 1, torch.tensor([6]))-torch.index_select(x, 1, torch.tensor([2])), torch.index_select(x, 1, torch.tensor([7]))-torch.index_select(x, 1, torch.tensor([3]))]])
    #np.matmul()
    return torch.cat((e1, e2), 1), torch.cat((dv1, dv2), 1)

def model_loss(x, net, device):

    i = 0
    u =net(x)
    
    return model_loss_(x, u, device)
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
        

    #print('X', len(X))
    #print('Y', len(Y))
    x_nan_index = np.isnan(X).any(axis=1)
    X = X[~x_nan_index]
    Y = Y[~x_nan_index]

    y_nan_index = np.isnan(Y).any(axis=1)
    X = X[~y_nan_index]
    Y = Y[~y_nan_index]

    dataset_size = len(X)
    validation_split = 0.2
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices = indices[split:]
    test_indices = indices[:split]
    X_test = X[test_indices]
    Y_test = Y[test_indices]
    X_train = X[train_indices]
    Y_train = Y[train_indices]
    #a_train = a[train_indices]
    train_validation = []
    test_validation = []
    #X= X[0:3]
    #Y = Y[0:3]
    net = Net()
    net = net.to(device)
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(net.parameters())
    
    ### (3) Training / Fitting
    iterations = 13000
    pt_x_bc = Variable(torch.from_numpy(X_train).float(), requires_grad=False).to(device)
    pt_y_bc = Variable(torch.from_numpy(Y_train).float(), requires_grad=False).to(device)

    pt_x_validation = Variable(torch.from_numpy(X_test).float(), requires_grad=False).to(device)
    pt_y_validation = Variable(torch.from_numpy(Y_test).float(), requires_grad=False).to(device)
    early_stop = EarlyStopper(5, 0.1, 0.00001)
    #previous_validation_loss = 99999999.0
    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero
        net_bc_out = net(pt_x_bc)
        mse_u = mse_cost_function(net_bc_out, pt_y_bc)
        e, dv = model_loss(pt_x_bc, net, device)
        mse_f = mse_cost_function(e, dv)
        loss = mse_u + mse_f

        #@torch.no_grad
        net_bc_out = net(pt_x_validation)
        mse_u = mse_cost_function(net_bc_out, pt_y_validation)
        e, dv = model_loss(pt_x_validation, net, device)
        mse_f = mse_cost_function(e, dv)
        valitation_loss = mse_u + mse_f

        loss.backward() # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

        train_validation.append(loss.cpu().data.numpy())
        test_validation.append(valitation_loss.cpu().data.numpy())

        if (early_stop.early_stop(valitation_loss.cpu().data.numpy())):
            print("training halt")
            break
        
        with torch.autograd.no_grad():
    	    print(epoch,"Traning Loss:",loss.data,",Validation Loss:", valitation_loss.data)
            

    np.save(os.path.join('.', 'out', 'train_deep.loss'), np.array(train_validation))
    np.save(os.path.join('.', 'out', 'test_deep.loss'), np.array(test_validation))


    print('saving model')
    torch.save(net.state_dict(), os.path.join('.', 'out', 'model_uxt_deepNeurons.pt'))

if __name__ == '__main__':
    main()