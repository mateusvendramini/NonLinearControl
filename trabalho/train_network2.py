import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils
import torch.utils.checkpoint
import torch_directml
import numpy as np
import os
from generate_training import array_folder
from normalize_inputs import  Normalize
x_offset_tensor = None
y_offset_tensor = None
x_mod_tensor = None
y_mod_tensor = None

network_file = 'net_adam.pt'
checkpoint_file = 'checkpoint.pt'
def map_m1 (num):
    return map_var(num, 2, 8 )
def map_m2(num):
    return map_var(num, 1, 5)
def map_l1(num):
    return map_var(num, 1, 2)
def map_l2(num):
    return map_var(num, 0.5, 1.5)
def map_I1(num):
    return map_var(num, 0.1, 0.4)
def map_I2(num):
    return map_var(num, 0.05, 0.2)
def map_F(num):
    return map_var(num, 10, 20)
def map_var (num, min, max):
    return num*(max-min) + min
#def unomalize(X, Y):
    

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
            self.low_dloss_counter = 0
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)
    
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         #self.layers = nn.Sequential()
#         self.hidden_layer1 = nn.Linear(16,96)
#         self.hidden_layer2 = nn.Linear(96,48)
#         self.hidden_layer3 = nn.Linear(48,32)
#         self.hidden_layer4 = nn.Linear(32,16)
#         self.output_layer = nn.Linear(16,8)
#         self.act = torch.relu
#         #self.output_layer = nn.Linear(12,8)

#     def forward(self, x):
#         #inputs = torch.cat([x,t],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
#         layer1_out = self.act(self.hidden_layer1(x))
#         layer2_out = self.act(self.hidden_layer2(layer1_out))
#         layer3_out = self.act(self.hidden_layer3(layer2_out))
#         layer4_out = self.act(self.hidden_layer4(layer3_out))
#         output =     torch.sigmoid(self.output_layer(layer4_out))
#         #output = torch.sigmoid(self.output_layer(layer5_out)) ## For regression, no activation is used in output layer
        return output
    
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def H_(m1, m2, L1, L2, I1, I2, F1, F2, q1, q2, q3, q4):
    g = 9.8
    L1_qr = L1*L1
    L1L2 = L1*L2
    L2_qr = L2*L2
    a1 = I1 + torch.mul(torch.mul(m1,L1_qr), 0.25)  
    #m1L1 = torch.mul(m1,L1_qr)  
    #a1 += torch.mul(m1L1, 0.25)
    b = L1_qr + torch.mul(0.25, L2_qr)
    a1 += torch.mul(m2, b)
    #a1 = I1 +(m1*L1*L1)/4 + m2*(L1*L1 + (L2*L2)/4)
    a2 = I2 +torch.mul(torch.mul(m2, L2_qr) , 0.25)
    a3 = m2*L1L2/2
    cosq2 = torch.cos(q2)
    #H = np.array([[a1+2*a3*np.cos(q2), a2+a3*np.cos(q2)],
    #                [a2+a3*np.cos(q2), a2]])
    H = a1+2*torch.mul(a3, cosq2), a2+a3*torch.cos(q2), a2+a3*torch.cos(q2), a2 #return H11, H12, H21, H22
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
    #procurar erro aqui
    F1_ = torch.mul(F1,q3)
    F2_ = torch.mul(F2,q4)

    C1_ = a4*torch.cos(q1)+a5*torch.cos(q1+q2)
    C2_ = a5*torch.cos(q1+q2)
    E1_ = -2*a3*torch.sin(q2)*q3*q4-a3*torch.sin(q2)*(q4*q4)
    E2_ = a3*torch.sin(q2)*(q3*q3)
    q1dd_den = F1_ + C1_ + E1_
    q2dd_den = F2_+ C2_ + E2_
    
    return (-q1dd_den, -q2dd_den)

# def f(u, x):
#     return f_(torch.index_select(u, 1, 0), torch.index_select(u, 1, 1), 
#              torch.index_select(u, 1, 2), torch.index_select(u, 1, 3),
#              torch.index_select(u, 1, 4), torch.index_select(u, 1, 5), 
#              torch.index_select(u, 1, 6), torch.index_select(u, 1, 7),
#              torch.index_select(x, 1, 0), torch.index_select(x, 1, 1), torch.index_select(x, 1, 2), torch.index_select(x, 1, 3))

def model_loss_(x, u_hat, device, norm):
    #formatoloss_(x, u_hat, device, norm):
    #formato de x
    #[[q1, q2, q3, q4, T1, T2, q3_nex, q4_next, parametros hat]
    

    #u_hat = net(x) # m1 m2 L1 L2 I1 I2 F1 F2 estimados
    #u_hat = u_hat.detach().numpy() 
    #u_hat = u_hat.numpy() 
    #x = x.numpy()
    x, u_hat = norm.unormalize(x, u_hat)
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

#todo inline talvez seja mais rápido ou talvez não

    m1x = map_m1(torch.index_select(x, 1, torch.tensor([8])).to(device))
    m2x = map_m2(torch.index_select(x, 1, torch.tensor([9])).to(device))
    L1x = map_l1(torch.index_select(x, 1, torch.tensor([10])).to(device))
    L2x = map_l2(torch.index_select(x, 1, torch.tensor([11])).to(device))
    I1x = map_I1(torch.index_select(x, 1, torch.tensor([12])).to(device))
    I2x = map_I2(torch.index_select(x, 1, torch.tensor([13])).to(device))
    F1x = map_F(torch.index_select(x, 1, torch.tensor([14])).to(device))
    F2x = map_F(torch.index_select(x, 1, torch.tensor([15])).to(device))
    
    m1u = map_m1(torch.index_select(u_hat, 1, torch.tensor([0])).to(device))
    m2u = map_m2(torch.index_select(u_hat, 1, torch.tensor([1])).to(device))
    L1u = map_l1(torch.index_select(u_hat, 1, torch.tensor([2])).to(device))
    L2u = map_l2(torch.index_select(u_hat, 1, torch.tensor([3])).to(device))
    I1u = map_I1(torch.index_select(u_hat, 1, torch.tensor([4])).to(device))
    I2u = map_I2(torch.index_select(u_hat, 1, torch.tensor([5])).to(device))
    F1u = map_F (torch.index_select(u_hat, 1, torch.tensor([6])).to(device))
    F2u = map_F (torch.index_select(u_hat, 1, torch.tensor([7])).to(device))

    
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
    Hi_next11, Hi_next12, Hi_next21, Hi_next22 = H_(m1x+m1u, m2x+m2u, L1x+L1u, L2x+L2u, I1x+I1u, I2x+I2u, F1x+F1u, F2x+F2u, q1, q2, q3, q4)
    #Hi11, Hi12, Hi21, Hi22 = H_(m1x, m2x, L1x, L2x, I1x, I2x, F1x, F2x, q1, q2, q3, q4)#.to(device) #q1, q2, q3, q4
    #fi1, fi2  = f_(m1x, m2x, L1x, L2x, I1x, I2x, F1x, F2x, q1, q2, q3, q4) # formato 
    fi_next1, fi_next2 = f_(m1x+m1u, m2x+m2u, L1x+L1u, L2x+L2u, I1x+I1u, I2x+I2u, F1x+F1u, F2x+F2u, q1, q2, q3, q4)#.to(device)

    #Hi = Hi.to(device)
    den = 1/(Hi_next11*Hi_next22 - Hi_next12*Hi_next21)
    Hi_next_inv11 = den * Hi_next22
    Hi_next_inv12 = -den * Hi_next12
    Hi_next_inv21 =-den * Hi_next21
    Hi_next_inv22 = den * Hi_next11

    #e = torch.matmul(Hi_next_inv,fi_next - fi) - torch.matmul(torch.matmul(Hi_next_inv, Hi), torch.tensor([[torch.index_select(x, 1, 4)], [torch.index_select(x, 1, 5)]])) 
    #HH11 = Hi_next_inv11*Hi11 + Hi_next_inv12*Hi21
    #HH12 = Hi_next_inv11*Hi12 + Hi_next_inv12*Hi22
    #HH21 = Hi_next_inv21*Hi11 + Hi_next_inv22*Hi21
    #HH22 =  Hi_next_inv21*Hi12 + Hi_next_inv22*Hi22

    e1 = (Hi_next_inv11*(-fi_next1+T1) + Hi_next_inv12*(-fi_next2+T2))*dt #- (HH11*T1 + HH12*T2))*dt
    e2 = (Hi_next_inv21*(-fi_next1+T1) + Hi_next_inv22*(-fi_next2+T2))*dt #- (HH21*T1 + HH22*T2))*dt
    dv1 = v1-q3 #- v1  
    dv2 = v2 - q4    
    #e = dt*e
    #e = torch.transpose(e)
    #dv = torch.tensor([[torch.index_select(x, 1, torch.tensor([6]))-torch.index_select(x, 1, torch.tensor([2])), torch.index_select(x, 1, torch.tensor([7]))-torch.index_select(x, 1, torch.tensor([3]))]])
    #np.matmul()
    return torch.cat((e1, e2), 1), torch.cat((dv1, dv2), 1)

def model_loss(x, net_out, device, norm):

    #i = 0
    #u =net(x)
    
    return model_loss_(x, net_out, device, norm)
def main():
    device = torch.device("cpu") #torch_directml.device()#torch.device("cpu")#torch_directml.device() #torch_directml.device()
    print('using device ', device)
    #print('device name', torch_directml.device_name(0))
    set_seed(666)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    Y = np.empty((0,8))  
    X = np.empty((0,16))
    #load data
    # for i in range (8):
    #     x_file_name='x_{0}_big.out.npy'.format(i)
    #     y_file_name='y_{0}_big.out.npy'.format(i)
    #     x_path = (os.path.join(array_folder, x_file_name))
    #     y_path = (os.path.join(array_folder, y_file_name))
    #     x = np.load(x_path, 'r')
    #     X = np.concatenate((X, x))
    #     y = np.load(y_path, 'r')
    #     Y = np.concatenate((Y, y))
    try:
        x_file_name='x_norm_big.out.npy'
        y_file_name='y_norm_big.out.npy'
        x_path = (os.path.join(array_folder, x_file_name))
        y_path = (os.path.join(array_folder, y_file_name))
        x = np.load(x_path, 'r')
        X = np.concatenate((X, x))
        y = np.load(y_path, 'r')
        Y = np.concatenate((Y, y))
    except Exception:
        print('fail to load model an directory ', array_folder)
        return -1

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
    X_train1 = X_train[len(X_train)//4:] #X_train#[len(X_train)//2:]
    Y_train1 = Y_train[len(Y_train)//4:]#Y_train#[len(Y_train)//2:]
    #X_train2 = X_train[:len(X_train)//2]
    #Y_train2 = Y_train[:len(Y_train)//2]
    
    #a_train = a[train_indices]
    train_validation = []
    test_validation = []
    #X= X[0:3]
    #Y = Y[0:3]
    net = Net()

    try:
        model_path=os.path.join(array_folder, checkpoint_file)
        checkpoint = torch.load(model_path, weights_only=False)
        net.load_state_dict(checkpoint['model_dict'])
        net = net.to(device)
        optimizer = torch.optim.Adam(params=net.parameters(), foreach=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        net.train()
    except Exception as e:
        print('Erro ao carregar ', e,e.args)
        print(str(e))
        net = net.to(device)
        optimizer = torch.optim.Adam(params=net.parameters(), foreach=True)

    #net = net.to(device)
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    #optimizer = torch.optim.Adam(net.parameters())
    
    ### (3) Training / Fitting
    iterations = 100
    loss_output = []
    pt_x_bc1 = torch.from_numpy(X_train1).float().to(device)
    #pt_x_bc.requires_grad = True
    pt_y_bc1 = torch.from_numpy(Y_train1).float().to(device)
    norm = Normalize(device)
    #pt_x_bc2 = torch.from_numpy(X_train2).float().to(device)
    #pt_x_bc.requires_grad = True
    #pt_y_bc2 = torch.from_numpy(Y_train2).float().to(device)
    
    #pt_x_validation = Variable(torch.from_numpy(X_test).float(), requires_grad=False).to(device)
    #pt_y_validation = Variable(torch.from_numpy(Y_test).float(), requires_grad=False).to(device)
    #early_stop = EarlyStopper(5, 0.1, 0.00001)
    #previous_validation_loss = 99999999.0
    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero
        net_bc_out = net(pt_x_bc1)#torch.utils.checkpoint.checkpoint(net.forward, pt_x_bc, use_reentrant=False)#net(pt_x_bc)
        mse_u = mse_cost_function(net_bc_out, pt_y_bc1)
        loss = mse_cost_function(net_bc_out, pt_y_bc1)
        e, dv = model_loss(pt_x_bc1, net_bc_out, device, norm)
        loss = mse_u + mse_cost_function(e, dv)
        #loss = mse_u + mse_f

        loss.backward() # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

        optimizer.zero_grad() # to make the gradients zero
        #net_bc_out = net(pt_x_bc2)#torch.utils.checkpoint.checkpoint(net.forward, pt_x_bc, use_reentrant=False)#net(pt_x_bc)
        #mse_u = mse_cost_function(net_bc_out, pt_y_bc2)
        #loss = mse_cost_function(net_bc_out, pt_y_bc2)
        #e, dv = model_loss(pt_x_bc2, net_bc_out, device)
        #loss = mse_u #+ mse_cost_function(e, dv)
        #loss = mse_u + mse_f

        #loss.backward() # This is for computing gradients using backward propagation
        #optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

        #@torch.no_grad
        # net_bc_out = net(pt_x_validation)
        # mse_u = mse_cost_function(net_bc_out, pt_y_validation)
        # e, dv = model_loss(pt_x_validation, net, device)
        # mse_f = mse_cost_function(e, dv)
        # valitation_loss = mse_u + mse_f

        #loss.backward() # This is for computing gradients using backward propagation
        #optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

        #train_validation.append(loss.cpu().data.numpy())
        #test_validation.append(valitation_loss.cpu().data.numpy())

        # if (early_stop.early_stop(valitation_loss.cpu().data.numpy())):
        #     print("training halt")
        #     break
        # if epoch % 100 == 0:
        #     print('saving model')
        #     loss_saved = loss.clone()
        #     loss_saved = loss_saved.detach().cpu()
        #     net_save = net.cpu()#.detach()
        #     #optimizer_saved = optimizer.cpu()
        #     torch.save({
        #     'loss' : loss_saved,
        #     'model_dict' : net_save.state_dict(),
        #     'optimizer_state_dict' : optimizer.state_dict(),
        #     }, os.path.join(array_folder, checkpoint_file))
        #     #loss.to(device)
        #     net.train()
        #     net.to(device)
        saved_loss = torch.clone(loss).cpu()
        loss_output.append(saved_loss.data.numpy())  
        with torch.autograd.no_grad():
    	    print(epoch,"Traning Loss:",loss.data)#,",Validation Loss:", valitation_loss.data)
          

    #np.save(os.path.join(array_folder, 'train_deep.loss'), np.array(train_validation))
    #np.save(os.path.join(array_folder,'test_deep.loss'), np.array(test_validation))

    #print('saving model')

    print('saving model')
    torch.save(net.state_dict(), os.path.join(array_folder, network_file))
    net = net.cpu()
    loss = loss.cpu()
    torch.save({
        'loss' : loss,
        'model_dict' : net.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
    }, os.path.join(array_folder, checkpoint_file))
    np.save(os.path.join(array_folder, 'loss_output_2'), np.array(loss_output))

    pt_x_validation = Variable(torch.from_numpy(X_test[0:10]).float(), requires_grad=False).to(device)
    pt_y_validation = Variable(torch.from_numpy(Y_test[0:10]).float(), requires_grad=False).to(device)
    Y_predicted = net.forward(pt_x_validation)
    for i in range (10):
        print(i, ' entry[\r\n', pt_x_validation[i],'] predicted\r\n',Y_predicted[i],'golden', pt_y_validation[i])


if __name__ == '__main__':
    main()