#!/usr/bin/env python3
# -*- coding: utf-8 -*-
############################# IMPORTS #############################
import pathlib, os, sys
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('.')

import torch
import numpy as np
from tqdm import tqdm
from scipy import integrate

# My modules
from modules.neural_networks import VFPG_ours, VFPG_theirs, VFPG_theirs_v2
from modules.loss_functions import loss_DKL, loss_MSE, loss_DKL_v2
from modules.aux_functions import count_params, show_layers, train_loop, histogram
from modules.plotters import loss_paths_plot_theirs
from modules.lagrangians import L_HO, L_double_well

############################# GENERAL PARAMETERS #############################
# General parameters
#dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dev = 'cpu'
which_net = 'theirs'
M = 2048 # training set (latent space) size
N = 32 # length of the paths
batch_size = 128 # batch size
Nc = batch_size # number of gaussian components
M_MC = 10

t_0 = 0.
t_f = 0.5
x_i = torch.tensor(0.).to(dev)
x_f = torch.tensor(1.).to(dev)
t = [torch.tensor(e) for e in np.linspace(t_0, t_f, N)]
h = t[1] - t[0]
potential = L_HO

# Neural network parameters
input_size = 2 if which_net == 'theirs' else 1 # dimension of the input 
nhid = 10 # number of hidden neurons
hidden_size = 10 # dimension of the LSTM hidden state vector
out_size = 3 # size of the LSTM output, y
num_layers = 1 # number of stacked LSTM layers
Dense = True # controls wether there is a Linear layer after the LSTM or not
if which_net == 'ours':
    # Input to LSTM: M sequences, each of length 1, elements of dim input_size
    z = 3 * (2*torch.rand(M, 1, 1) - torch.ones(M, 1, 1)).to(dev)
elif which_net == 'theirs':
    # Input to LSTM: M sequences, each of length N, elements of dim input_size
    # z = torch.randn(M, N, input_size).to(dev) # everything is different
    z = [torch.randn(input_size).repeat(N, 1) for _ in range(M)]
    z = torch.stack(z).to(dev) # same 2D for all steps, different 2D for different paths
    
# Hyperparameters
learning_rate = 1e-4

# Training parameters
n_epochs = 3000

# Plotting parameters
show_periodic_plots = True
n_plots = n_epochs
bound = 20
show_loss_i = True
show_loss_f = True
dx = 0.1
leap = n_epochs/n_plots

# Saves
save_model = False

torch.manual_seed(1)
dx_list = [dx for e in range(M)]

############################# NEURAL NETWORK #############################
hidden_size = hidden_size if Dense else out_size
if which_net == 'ours':
    vfpg = VFPG_ours(dev=dev,
                     M=batch_size,
                     N=N, 
                     input_size=input_size, 
                     nhid=nhid,
                     hidden_size=hidden_size, 
                     out_size=out_size,
                     num_layers=num_layers, 
                     Dense=Dense).to(dev)
elif which_net == 'theirs':
    vfpg = VFPG_theirs_v2(dev=dev,
                       batch_size=batch_size,
                       N=N, 
                       input_size=input_size, 
                       nhid=nhid,
                       hidden_size=hidden_size, 
                       out_size=out_size,
                       num_layers=num_layers, 
                       Dense=Dense).to(dev)
    
optimizer = torch.optim.Adam(params=vfpg.parameters(), 
                             lr=learning_rate)
loss_fn = loss_DKL_v2

############################# EPOCH LOOP #############################
loss_list = []
loss_KL_list = [] 
loss_i_list = []
loss_f_list = [] 
print(f'Training a model with {count_params(vfpg)} parameters on {dev}.\n')

for j in tqdm(range(n_epochs)):
    for b in range(M // batch_size):
        z_b = z[b*batch_size:(b+1)*batch_size, :, :]
        L, L_KL, L_i, L_f, delta_L, paths = train_loop(model=vfpg, 
                                                       loss_fn=loss_fn, 
                                                       optimizer=optimizer, 
                                                       train_set=z_b, 
                                                       target_data=0.,
                                                       potential=potential,
                                                       M_MC=M_MC,
                                                       h=h,
                                                       x_i=x_i,
                                                       x_f=x_f)
        
    # Loss tracking
    loss_list.append(L.item())
    loss_KL_list.append(L_KL.item())
    loss_i_list.append(L_i.item())
    loss_f_list.append(L_f.item())
    # Periodic plots
    if (j+1) % leap == 0 and show_periodic_plots:

        # We compute the wave function from the paths sampled by the LSTM
        counts = list(map(histogram, paths.cpu().detach().numpy(), dx_list))
        wf = np.sum(counts, axis=0)
        wf_norm = integrate.simpson(y=wf, x=np.linspace(-4.95, 4.95, 100))
        wf /= wf_norm
                                                  
        # Plotting
        loss_paths_plot_theirs(bound=bound,
                        time_grid=t, 
                        path_manifold=paths.cpu(), 
                        wf=wf,
                        current_epoch=j, 
                        loss_list=loss_list,
                        loss_KL_list=loss_KL_list,
                        loss_i_list=loss_i_list,
                        loss_f_list=loss_f_list,
                        delta_L=delta_L,
                        show_loss_i=show_loss_i,
                        show_loss_f=show_loss_f)
        
print('\nDone! :)')

# Save the model
if save_model:
    state = {
        'epoch': n_epochs,
        'state_dict': vfpg.state_dict(),
        'optimizer': optimizer.state_dict()
        }
    model_name = 'first_models.pt'
    torch.save(state, model_name)
    print(f'Model correctly saved at: {model_name}')
    
"""
model.load_state_dict(state['state_dict'])
optimizer.load_state_dict(state['optimizer'])
"""









