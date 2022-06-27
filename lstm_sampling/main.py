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
from modules.neural_networks import VFPG_ours, VFPG_theirs
from modules.loss_functions import loss_DKL, loss_MSE
from modules.aux_functions import count_params, show_layers, train_loop
from modules.plotters import loss_paths_plot

############################# GENERAL PARAMETERS #############################
# General parameters
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
which_net = 'theirs'
M = 1 # number of paths (for Monte Carlo estimates)
N = 5 # length of the path
Nc = 5 # number of gaussian components
mus, sigmas = torch.zeros(N), torch.ones(N)

# Neural network parameters
input_size = 2 if which_net == 'theirs' else 1 # dimension of the input 
nhid = 6 # number of hidden neurons
hidden_size = 7 # dimension of the LSTM hidden state vector
out_size = (1 + 2)*Nc # size of the LSTM output, y
num_layers = 1 # number of stacked LSTM layers
Dense = True # controls wether there is a Linear layer after the LSTM or not
if which_net == 'ours':
    # Input to LSTM: M sequences, each of length 1, elements of dim input_size
    z = 5 * (2*torch.rand(M, 1, 1)-torch.ones(M, 1, 1)).to(dev)
elif which_net == 'theirs':
    # Input to LSTM: M sequences, each of length N, elements of dim input_size
    # z = torch.randn(M, N, input_size).to(dev) # everything is different
    z = [5*torch.randn(input_size).repeat(N, 1) for _ in range(M)]
    z = torch.stack(z).to(dev) # same 2D for all steps, different 2D for different paths
    
# Hyperparameters
learning_rate = 1e-5
epsilon = 1e-8
smoothing_constant = 0.9

# Training parameters
n_epochs = 10000

# Plotting parameters
n_plots = 5
leap = n_epochs/n_plots
bound = 10
show_periodic_plots = True

# Saves
save_model = False

torch.manual_seed(1)

############################# NEURAL NETWORK #############################
hidden_size = hidden_size if Dense else out_size
if which_net == 'ours':
    vfpg = VFPG_ours(dev=dev,
                     M=M,
                     N=N, 
                     input_size=input_size, 
                     nhid=nhid,
                     hidden_size=hidden_size, 
                     out_size=out_size,
                     num_layers=num_layers, 
                     Dense=Dense).to(dev)
elif which_net == 'theirs':
    vfpg = VFPG_theirs(dev=dev,
                       M=M,
                       N=N, 
                       input_size=input_size, 
                       nhid=nhid,
                       hidden_size=hidden_size, 
                       out_size=out_size,
                       num_layers=num_layers, 
                       Dense=Dense).to(dev)
    
optimizer = torch.optim.Adam(params=vfpg.parameters(), 
                             lr=learning_rate, 
                             eps=epsilon)
loss_fn = loss_DKL

############################# EPOCH LOOP #############################
loss_KL_list = []  
print(f'Training a model with {count_params(vfpg)} parameters on {dev}.\n')

for j in tqdm(range(n_epochs)):
    # Train loop
    #print('\n')
    #print(f'\nInput to LSTM at epoch {j}: {z}', z.size())
    L, delta_L, paths = train_loop(model=vfpg, 
                                   train_set=z,
                                   loss_fn=loss_fn, 
                                   optimizer=optimizer, 
                                   mus=mus,
                                   sigmas=sigmas)
    
    # Loss tracking
    loss_KL_list.append(L.item())
    
    # Periodic plots
    if (j+1) % leap == 0 and show_periodic_plots:                                                  
        loss_paths_plot(bound=bound,
                        path_manifold=paths.cpu(), 
                        current_epoch=j, 
                        loss_list=loss_KL_list,
                        delta_L=delta_L)
        
print('\nDone! :)')

# Save the model
if save_model:
    state = {
        'epoch': n_epochs,
        'state_dict': vfpg.state_dict(),
        'optimizer': optimizer.state_dict()
        }
    model_name = 'saved_models/model.pt'
    torch.save(state, model_name)
    print(f'Model correctly saved at: {model_name}')
    
"""
model.load_state_dict(state['state_dict'])
optimizer.load_state_dict(state['optimizer'])
"""









