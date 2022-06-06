######################## IMPORTS ########################
import numpy as np
from tqdm import tqdm
import torch,math,os

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
t_0 = 0.
t_f = 100.
eps = 5 # Maximum MC allowed error

# Names of files, directories
paths_file = f'../MonteCarlo/saved_data/paths_N{N}_M{M}.txt'
actions_file = f'../MonteCarlo/saved_data/actions_N{N}_M{M}.txt'
trained_models_path = 'saved_models/ff'
trained_plots_path = 'saved_plots/ff'

# Training parameters
n_epochs = 5000
Nhid = 30
num_layers = 2
batch_size = 1000
learning_rate = 1e-2
epsilon = 1e-8
smoothing_constant = 0.9
seed = 10

# Plotting parameters
adaptive_factor = 2.5
leap = n_epochs/10

# Saves/Booleans
save_model = False
save_plot = False
show_periodic_plots = True
mini_batching = False
continue_from_last = False

model_name = f'nhid{Nhid}_lr{learning_rate}_nlayers{num_layers}_seed{seed}'
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

<<<<<<< HEAD:supervised/feed_forward.py
=======
x_tensor = torch.cat([i.view(N,1,-1) for i in x_tensor])
>>>>>>> 4415f66b2f2d1fb3fcd4d8c43d7cbd3bbdfd5d84:feed_forward/feed_forward.py
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
    global params_mu,params_sigma
    
<<<<<<< HEAD:supervised/feed_forward.py
    #print(train_data.size(),train_data.dim())
=======
>>>>>>> 4415f66b2f2d1fb3fcd4d8c43d7cbd3bbdfd5d84:feed_forward/feed_forward.py
    output = q_params_nn(train_data)
    logq_tensor,params_mu,params_sigma = output[0],output[1],output[2]
    
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
<<<<<<< HEAD:supervised/feed_forward.py
q_params_nn = q_phi_ff(Nin,Nhid,Nout,num_layers=num_layers)
=======
q_params_nn = q_phi_layers(Nin,Nhid,Nout,num_layers=num_layers)
q_params_nn = q_phi_rnn(N=N,input_size=1,hidden_size=2*N,num_layers=1,Dense=False)
>>>>>>> 4415f66b2f2d1fb3fcd4d8c43d7cbd3bbdfd5d84:feed_forward/feed_forward.py
loss_fn = loss
optimizer = torch.optim.RMSprop(params=q_params_nn.parameters(),lr=learning_rate,eps=epsilon)

if continue_from_last:
    q_params_nn.load_state_dict(torch.load(f'{trained_models_path}{model_name}')['model_state_dict'])
    q_params_nn.eval()
    optimizer.load_state_dict(torch.load(f'{trained_models_path}model.pt')['optimizer_state_dict'])
    print('Resuming training from savepoint...\n')

######################## EPOCH LOOP ########################
x_axis,loss_list = [],[]
if not os.path.exists(trained_models_path+model_name+'.pt'):
    if mini_batching:
        for t in tqdm(range(n_epochs)):
            for b in range(M//batch_size):
                x_tensor_batch = x_tensor[b*batch_size:(b+1)*batch_size]
                train_loop(x_tensor_batch,loss_fn,optimizer)
                loss_list.append(I.item())
                x_axis.append(t)
                if t == n_epochs-1 or (t+1)%leap==0:
                    if show_periodic_plots:
                        color = 'blue' if error<=eps/100 else 'red'
                        loss_plot(x_axis,loss_list,x_tensor,params_mu,params_sigma,I.item(),adaptive_factor,error,color,save=False)
                if save_plot and t == n_epochs-1:
                    dir_support([trained_plots_path])
                    loss_plot(x_axis,loss_list,x_tensor,params_mu,params_sigmaI.item(),adaptive_factor,error,color,save=True,
                              model_name=trained_plots_path+model_name+'.png')
    else:
        for t in tqdm(range(n_epochs)):
            train_loop(x_tensor,loss_fn,optimizer)
            loss_list.append(I.item())
            x_axis.append(t)
            if t == n_epochs-1 or (t+1)%leap==0:
                if show_periodic_plots:
                    color = 'blue' if error<=eps/100 else 'red'
                    loss_plot(x_axis,loss_list,x_tensor,params_mu.detach(),params_sigma.detach(),I.item(),adaptive_factor,error,color,save=False)
            if save_plot and t == n_epochs-1:
                dir_support([trained_plots_path])
                loss_plot(x_axis,loss_list,x_tensor,params_mu,params_sigma,I.item(),adaptive_factor,error,color,save=True,
                          model_name=trained_plots_path+model_name+'.png')
    print('\nDone! :)')
    if save_plot:
        print(f'Plot saved in {trained_plots_path}')
        
    if save_model:
        dir_support([trained_models_path])
        state_dict = {'model_state_dict':q_params_nn.state_dict(),
                     'optimizer_state_dict':optimizer.state_dict(),
                     'epochs':n_epochs,
                     'Nin':Nin,
                     'Nhid':Nhid,
                     'Nout':Nout}
        torch.save(state_dict,trained_models_path+model_name+'.pt')
        print(f'Model saved in {trained_models_path}')
    
else:
    print(f'Model {model_name} has already been trained. Skipping...')
    
    
    
    