######################## IMPORTS ########################
import numpy as np
from tqdm import tqdm
import torch,math

# My modules
from modules.neural_networks import *
from modules.plotters import loss_plot
from modules.integration import simpson_weights
from modules.aux_functions import train_loop
from modules.physical_constants import *
from modules.dir_support import dir_support

######################## PARAMETERS ########################
# General parameters
N = 50
M = 10000
Nhid = 100
n_epochs = 100000
t_0 = 0.
t_f = 100.
paths_file = f'../MonteCarlo/saved_data/paths_N{N}_M{M}.txt'
actions_file = f'../MonteCarlo/saved_data/actions_N{N}_M{M}.txt'
trained_models_path = 'saved_models/'
leap = n_epochs/100
seed = 5
save_model = True
show_periodic_plots = True
mini_batching = False
batch_size = 1000
continue_from_last = True
epsilon = 5 # percent

# Training parameters
learning_rate = 1e-3
epsilon = 1e-8
smoothing_constant = 0.9

# Plotting parameters
adaptive_factor = 0.

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
def loss(train_data):
    """
    Loss function.

    Returns
    -------
    loss : tensor
        Kullback-Leibler divergence.

    """
    global I,I2,error,logq_tensor
    
    logq_tensor = q_params_nn(train_data)
    
    # Monte Carlo integration 
    M = len(train_data)
    I = -(1/M)*torch.sum((1/hbar)*S_tensor[:M]+logq_tensor)
    I2 = (1/M)*torch.sum((logq_tensor+(1/hbar)*S_tensor[:M])**2)
    error = (1/math.sqrt(M))*torch.sqrt(I2-I**2)
    
    return I
   
######################## NN STUFF ########################
Nin = N
Nout = Nin
torch.manual_seed(seed)
W1 = torch.rand(Nhid,Nin,requires_grad=True)*(-1.)
B = torch.rand(Nhid)*2.-torch.tensor(1.)
W2 = torch.rand(Nout,Nhid,requires_grad=True) 
#q_params_nn = q_phi(Nin,Nhid,Nout,W1,W2,B)
q_params_nn = q_phi_layers(Nin,Nhid,Nout,num_layers=2)
loss_fn = loss
optimizer = torch.optim.RMSprop(params=q_params_nn.parameters(),lr=learning_rate,eps=epsilon)

if continue_from_last:
    q_params_nn.load_state_dict(torch.load(f'{trained_models_path}model.pt')['model_state_dict'])
    q_params_nn.eval()
    optimizer.load_state_dict(torch.load(f'{trained_models_path}model.pt')['optimizer_state_dict'])
    print('Resuming training from savepoint...\n')

######################## EPOCH LOOP ########################
x_axis,loss_list = [],[]
if mini_batching:
    for t in tqdm(range(n_epochs)):
        for b in range(M//batch_size):
            x_tensor_batch = x_tensor[b*batch_size:(b+1)*batch_size]
            train_loop(x_tensor_batch,loss_fn,optimizer)
            loss_list.append(I.item())
            x_axis.append(t)
            if t == n_epochs-1 or (t+1)%leap==0:
                #print(error.item())
                if show_periodic_plots:
                    loss_plot(x=x_axis,y=loss_list)
else:
    for t in tqdm(range(n_epochs)):
        train_loop(x_tensor,loss_fn,optimizer)
        loss_list.append(I.item())
        x_axis.append(t)
        if t == n_epochs-1 or (t+1)%leap==0:
            #print(I.item())
            if show_periodic_plots:
                color = 'blue' if error<=eps/100 else 'red'
                loss_plot(x_axis,loss_list,I.item(),adaptive_factor,color)
print('\nDone! :)')
    
if save_model:
    dir_support([trained_models_path])
    state_dict = {'model_state_dict':q_params_nn.state_dict(),
                 'optimizer_state_dict':optimizer.state_dict(),
                 'Nin':Nin,
                 'Nhid':Nhid,
                 'Nout':Nout}
    torch.save(state_dict,trained_models_path+'model.pt')
    print(f'Model saved in {trained_models_path}')
    
    
    
    
    
    
    
    