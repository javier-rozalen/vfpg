#!/usr/bin/env python3
# -*- coding: utf-8 -*-
############################# IMPORTS #############################
import pathlib, os, sys
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('.')

import torch, math, time
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.autograd import grad
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

# My modules
#from modules.physical_constants import *
from modules.neural_networks import VFPG, show_layers, count_params
from modules.loss_functions import loss_DKL
from modules.aux_functions import train_loop
from modules.lagrangians import L_HO
from modules.plotters import loss_paths_plot

############################# GENERAL PARAMETERS #############################
# General parameters
M = 5 # number of paths (for Monte Carlo estimates)
N = 4 # length of the path
Nc = 3 # number of gaussian components

x_i = 0.
x_f = 0.
t_0 = 0.
t_f = 100.
t = [torch.tensor(e) for e in np.linspace(t_0, t_f, N)]
h = t[1]-t[0]

# Neural network parameters
input_size = 1 # dimension of the input 
nhid = 6 # number of hidden neurons
hidden_size = 7 # dimension of the LSTM hidden state vector
out_size = (input_size + 2)*Nc # size of the LSTM output, y
num_layers = 1 # number of stacked LSTM layers
Dense = True # controls wether there is a Linear layer after the LSTM or not
# Input to LSTM: M sequences, each of length 1, elements of dim input_size
z = 6.*torch.rand(M, 1, input_size)-torch.tensor(3.).repeat(M, 1, input_size)
#z = torch.zeros(M, 1, input_size)

# Hyperparameters
learning_rate = 1e-3
epsilon = 1e-8
smoothing_constant = 0.9

# Training parameters
n_epochs = 5000
mini_batching = False

# Plotting parameters
n_plots = 50
leap = n_epochs/n_plots
bound = 10
show_periodic_plots = True

# Saves
save_model = False

torch.manual_seed(1)

############################# NEURAL NETWORK #############################
hidden_size = hidden_size if Dense else out_size
vfpg = VFPG(M=M,
            N=N, 
            input_size=input_size, 
            nhid=nhid,
            hidden_size=hidden_size, 
            out_size=out_size,
            num_layers=num_layers, 
            Dense=Dense)    

optimizer = torch.optim.Adam(params=vfpg.parameters(), 
                                lr=learning_rate, 
                                eps=epsilon)

loss_fn = loss_DKL

############################# EPOCH LOOP #############################
loss_KL_list = [] 

for j in tqdm(range(n_epochs)):
    #print('\n')
    # Train loop
    #Lprint(f'\nInput to LSTM at epoch {j}: {z}', z.size())
    L, delta_L, paths = train_loop(model=vfpg, 
                                   loss_fn=loss_fn, 
                                   optimizer=optimizer, 
                                   train_set=z, 
                                   h=h)
    
    # Loss tracking
    loss_KL_list.append(L.item())
    
    # Periodic plots
    if (j+1)%leap == 0 and show_periodic_plots:
        loss_paths_plot(bound=bound,
                        time_grid=t, 
                        path_manifold=paths, 
                        current_epoch=j, 
                        loss_list=loss_KL_list,
                        delta_L = delta_L)
        
print(f'Done! :)')

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









