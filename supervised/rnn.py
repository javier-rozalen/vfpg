######################## IMPORTS ########################
import pathlib, os, sys
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('.')

import numpy as np
from tqdm import tqdm
import torch,math

# My modules
from modules.neural_networks import *
from modules.plotters import loss_plot,loss_plot_rnn
from modules.integration import simpson_weights
from modules.aux_functions import train_loop
from modules.physical_constants import *
from modules.dir_support import dir_support

######################## PARAMETERS ########################
# General parameters
N = 15
M = 10000
t_0 = 0.
t_f = 100.

# Names of files, directories
paths_file = f'../MonteCarlo/saved_data/paths_N{N}_M{M}.txt'
actions_file = f'../MonteCarlo/saved_data/actions_N{N}_M{M}.txt'
trained_models_path = 'saved_models/rnn/'
trained_plots_path = 'saved_plots/rnn/'

# Training parameters
pdf = 'iidg' # 'gmm', 'iidg'
n_epochs = 5000
input_size = 1
num_layers = 2
learning_rate = 1e-3
epsilon = 1e-8
smoothing_constant = 0.9
seed = 1

# Plotting parameters
adaptive_factor = 0. # Set to 0. when no adaptive limits are wanted
leap = n_epochs/20
eps = 5 # Maximum MC allowed error

# Saves/Booleans
Dense = True
save_model = True
save_plot = True
show_periodic_plots = True
mini_batching = False
continue_from_last = False

model_name = f'N{N}_M{M}_num_layers{num_layers}_Dense{Dense}_lr{learning_rate}_seed{seed}'
######################## DATA FETCHING ########################
print('Fetching data...')
path_manifold,S_manifold = [],[]
with open(paths_file,'r') as file:
    for line in file.readlines():
        path = [[float(x)] for x in line.split(' ')]
        path_manifold.append(torch.tensor(path))
    file.close()
with open(actions_file,'r') as file:
    for line in file.readlines():
        S = torch.tensor(float(line))
        S_manifold.append(S)
    file.close()
path_manifold = path_manifold[:M]
S_manifold = S_manifold[:M]
x_tensor = torch.stack(path_manifold)
S_tensor = torch.stack(S_manifold)
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
    global params_mu,params_sigma
        
    logq_tensor,params_mu,params_sigma = q_params_nn(train_data)
    
    # Monte Carlo integration 
    M = len(train_data)
    I = -(1/M)*torch.sum((1/hbar)*S_tensor+logq_tensor.squeeze())
    I2 = (1/M)*torch.sum((logq_tensor.squeeze()+(1/hbar)*S_tensor)**2)
    error = (1/math.sqrt(M))*torch.sqrt(I2-I**2)
    
    return I
   
######################## NN STUFF ########################
torch.manual_seed(seed)
hidden_size = 2*input_size if pdf=='iidg' else 3*input_size
q_params_nn = q_phi_rnn(input_size,hidden_size,num_layers,Dense)
### Explanation of the inputs
# N --> length of the sequence (path)
# input_size --> number of features / dimensions of each element of the sequence (path)
# hidden_size --> dimensions of the output of the LSTM
# num_layers --> number of layers for each gate
# Dense --> whether to add a Linear layer on top of each gate or not

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
                        loss_plot_rnn(x_axis,loss_list,I.item(),adaptive_factor,100*error,color,save=False)
                if save_plot and t == n_epochs-1:
                    dir_support([trained_plots_path])
                    loss_plot_rnn(x_axis,loss_list,I.item(),adaptive_factor,100*error,color,save=True,
                              model_name=trained_plots_path+model_name+'.png')
    else:
        for t in tqdm(range(n_epochs)):
            train_loop(x_tensor,loss_fn,optimizer)
            loss_list.append(I.item())
            x_axis.append(t)
            if t == n_epochs-1 or (t+1)%leap==0:
                if show_periodic_plots:
                    color = 'blue' if error<=eps/100 else 'red'
                    loss_plot_rnn(x_axis,loss_list,I.item(),adaptive_factor,100*error,color,save=False)
            if save_plot and t == n_epochs-1:
                dir_support([trained_plots_path])
                loss_plot_rnn(x_axis,loss_list,I.item(),adaptive_factor,100*error,color,save=True,
                          model_name=trained_plots_path+model_name+'.png')
    print('\nDone! :)')
    if save_plot:
        print(f'Plot saved in {trained_plots_path}')
        
    if save_model:
        dir_support([trained_models_path])
        state_dict = {'model_state_dict':q_params_nn.state_dict(),
                     'optimizer_state_dict':optimizer.state_dict(),
                     'epochs':n_epochs,
                     'M':M,
                     'N':N,
                     'input_size':input_size,
                     'hidden_size':hidden_size,
                     'num_layers':num_layers,
                     'Dense':Dense}
        torch.save(state_dict,trained_models_path+model_name+'.pt')
        print(f'Model saved in {trained_models_path}')
    
else:
    print(f'Model {model_name} has already been trained. Skipping...')
    
    
    
    