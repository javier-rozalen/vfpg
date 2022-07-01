# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 11:05:56 2022

@author: javir
"""

######################## IMPORTS ########################
# Change to the directory of this script
import pathlib, os, sys
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('.')

# General modules
import numpy as np
from tqdm import tqdm
import torch, math

# My modules
from modules.neural_networks import VAE
from modules.plotters import loss
from modules.aux_functions import *
from modules.physical_constants import *
from modules.loss_functions import ELBO

######################## PARAMETERS ########################
# General parameters
dev = 'cpu'
N = 30
M = 10000

# Names of files, directories
paths_file = f'../MonteCarlo/saved_data/paths_N{N}_M{M}.txt'
actions_file = f'../MonteCarlo/saved_data/actions_N{N}_M{M}.txt'
trained_models_path = 'saved_models/'
trained_plots_path = 'saved_plots/'

# Neural network parameters
latent_size = 10
hidden_size_enc = 20
hidden_size_dec = 20

# Training parameters
n_epochs = 3000
batch_size = 100
lr = 1e-3
seed = 1

# Plotting parameters
adaptive_factor = 2.5
leap = n_epochs/20

# Saves/Booleans
save_model = False
save_plot = False
show_periodic_plots = True

trained_model_name = 'vae.pt'

torch.manual_seed(seed)
######################## DATA FETCHING ########################
print('Fetching data...')
train_set, actions_set = fetch_data(M, paths_file, actions_file)
print('Data fetching complete.\n')

######################## NEURAL NETWORK ########################
vae = VAE(sample_size=N,
          latent_size=latent_size,
          hidden_size_enc=hidden_size_enc,
          hidden_size_dec=hidden_size_dec)

loss_fn = ELBO
optimizer = torch.optim.Adam(params=vae.parameters(),
                             lr=lr)

######################## TRAINING LOOP ########################
loss_list = []
MC_error_list = []
# Epoch loop
for j in tqdm(range(n_epochs)):
    # Batch loop
    for b in range(M // batch_size):
        loss, d_loss = train_loop(model=vae,
                                  loss_fn=loss_fn,
                                  optimizer=optimizer,
                                  train_set=train_set)
    loss_list.append(loss)
    
    if (j % leap) == 0 and show_periodic_plots:
        loss_paths_wf(bound, time_grid, path_)

    if j == n_epochs - 1 and save_plot:
        loss()

print('Done! :)')

# We save the model
if save_model:
    dir_support(trained_models_path)
    state = {
        'epoch': n_epochs,
        'state_dict': vae.state_dict(),
        'optimizer': optimizer.state_dict()
        }
    torch.save(state, trained_model_name)
    print(f'Model correctly saved at: {trained_model_name}')
























