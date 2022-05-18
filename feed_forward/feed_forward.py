######################## IMPORTS ########################
import numpy as np
from tqdm import tqdm
import torch,math

# My modules
from modules.neural_networks import q_phi
from modules.plotters import loss_plot
from modules.integration import simpson_weights
from modules.aux_functions import train_loop
from modules.physical_constants import *

######################## PARAMETERS ########################
# General parameters
Nhid = 100
n_epochs = 100
t_0 = 0.
t_f = 100.
paths_file = '../MonteCarlo/saved_data/paths.txt'
actions_file = '../MonteCarlo/saved_data/actions.txt'

# Training parameters
learning_rate = 1e-2
epsilon = 1e-8
smoothing_constant = 0.9

######################## DATA FETCHING ########################
print('Fetching data...')
path_manifold,S_manifold = [],[]
with open(paths_file,'r') as file:
    for line in file.readlines():
        path = [torch.tensor(float(x)) for x in line.split(' ')]
        path_manifold.append(torch.stack(path))
    file.close()
with open(actions_file,'r') as file:
    for line in file.readlines():
        S = torch.tensor(float(line))
        S_manifold.append(S)
    file.close()
x_tensor = torch.stack(path_manifold)
S_tensor = torch.stack(S_manifold)
M = len(path_manifold)
N = len(path_manifold[0])
print('Data fetching complete.\n')

######################## LOSS FUNCTION ########################
def loss():
    """
    Loss function.

    Returns
    -------
    loss : tensor
        Kullback-Leibler divergence.

    """
    global I,I2,error
    
    q_manifold=[0]*M
    c = 0
    for x in path_manifold:
        q_x = q_params_nn(x)
        print(q_x)
        q_manifold[c]=q_x
        c+=1
    q_tensor = torch.stack(q_manifold)
    
    # Monte Carlo integration 
    I = (1/M)*torch.sum((1/hbar)*S_tensor+torch.log(q_tensor))
    #I2 = (1/M)*torch.sum((torch.log(q_tensor)+(1/hbar)*S_tensor)**2)
    #error = (1/math.sqrt(M))*torch.sqrt(I2-I**2)
    
    return I

######################## NN STUFF ########################
Nin = N
Nout = 2*Nin
W1 = torch.rand(Nhid,Nin,requires_grad=True)*(-1.)
B = torch.rand(Nhid)*2.-torch.tensor(1.)
W2 = torch.rand(Nout,Nhid,requires_grad=True) 
q_params_nn = q_phi(Nin,Nhid,Nout,W1,W2,B)
loss_fn = loss
optimizer = torch.optim.RMSprop(params=q_params_nn.parameters(),lr=learning_rate,eps=epsilon)

######################## EPOCH LOOP ########################
x_axis,loss_list = [],[]
for t in tqdm(range(n_epochs)):
    train_loop(loss_fn,optimizer)
    loss_list.append(I.item())
    x_axis.append(t)
    print(x_axis,loss_list)
    loss_plot(x=x_axis,y=loss_list)
    
    
    
    
    
    
    
    
    