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
from modules.dir_support import dir_support

######################## PARAMETERS ########################
# General parameters
Nhid = 100
n_epochs = 1000
t_0 = 0.
t_f = 100.
paths_file = '../MonteCarlo/saved_data/paths.txt'
actions_file = '../MonteCarlo/saved_data/actions.txt'
trained_models_path = 'saved_models/'
leap = n_epochs/20
seed = 1
save_model = True
show_periodic_plots = True

# Training parameters
learning_rate = 1e-5
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
    global I,I2,error,logq_tensor
    
    logq_tensor = q_params_nn(x_tensor)
    
    # Monte Carlo integration 
    I = -(1/M)*torch.sum((1/hbar)*S_tensor+logq_tensor)
    I2 = (1/M)*torch.sum((logq_tensor+(1/hbar)*S_tensor)**2)
    error = (1/math.sqrt(M))*torch.sqrt(I2-I**2)
    
    return I
   

######################## NN STUFF ########################
Nin = N
Nout = Nin
torch.manual_seed(seed)
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
    if t == n_epochs-1 or (t+1)%leap==0:
        #print(I.item())
        if show_periodic_plots:
            loss_plot(x=x_axis,y=loss_list)
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
    
    
    
    
    
    
    
    