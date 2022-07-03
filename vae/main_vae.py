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
from modules.plotters import loss_plot
from modules.aux_functions import *
from modules.physical_constants import *
from modules.loss_functions import ELBO

######################## PARAMETERS ########################
# General parameters
dev = 'cpu'
N = 10
M = 10000

# Names of files, directories
paths_file = f'../MonteCarlo/saved_data/paths_N{N}_M{M}.txt'
actions_file = f'../MonteCarlo/saved_data/actions_N{N}_M{M}.txt'
trained_models_path = 'saved_models/'
trained_plots_path = 'saved_plots/'

# Neural network parameters
latent_size = 5
hidden_size_enc = 100
hidden_size_dec = 100

# Training parameters
n_epochs = 3
batch_size = 500
MC_size = 500
lr = 1e-4

# Plotting parameters
n_plots = 10
adaptive_factor = 2.5
leap = n_epochs/n_plots

# Saves/Booleans
save_model = True
save_plot = True
show_periodic_plots = True

trained_model_name = (f'nepochs{n_epochs}_lr{lr}_N{N}_n{MC_size}_b{batch_size}_'
                      f's{latent_size}')
full_model_name = trained_models_path + trained_model_name + '.pt'
full_plot_name = trained_plots_path + trained_model_name + '.pdf'

######################## DATA FETCHING ########################
print('Fetching data...')
train_set, actions_set = fetch_data(M, paths_file, actions_file)
print('Data fetching complete.\n')

M = 1000
train_set = train_set[:M]
# train_set shape: [M, N]
######################## NEURAL NETWORK ########################
vae = VAE(sample_size=N,
          batch_size=batch_size,
          latent_size=latent_size,
          MC_size=MC_size,
          hidden_size_enc=hidden_size_enc,
          hidden_size_dec=hidden_size_dec)

loss_fn = ELBO
optimizer = torch.optim.Adam(params=vae.parameters(),
                             lr=lr)

######################## TRAINING LOOP #########################
print(f'Training a model with {count_params(vae)} parameters on {dev}.')
print(f'Sample size: {N}')
print(f'Latent size: {latent_size}')
print(f'Encoder hidden size: {hidden_size_enc}')
print(f'Decoder hidden size: {hidden_size_dec}')
print(f'MC_size: {MC_size}')
print(f'batch_size: {batch_size}')
print(f'lr: {lr}')
print(f'save_model: {save_model}')
print(f'save_plot: {save_plot}')
print('\n')

loss_list = []

# Epoch loop
for j in tqdm(range(n_epochs)):
    # Batch loop
    for b in range(M // batch_size):
        z_b = train_set[b*batch_size:(b+1)*batch_size, :]
        loss, MC_error = train_loop(model=vae,
                                    loss_fn=loss_fn,
                                    optimizer=optimizer,
                                    train_set=z_b)
    loss_list.append(loss)
    
    if (j % leap) == 0 and show_periodic_plots:
        loss_plot(loss_list=loss_list, 
                  MC_error=MC_error, 
                  current_epoch=j, 
                  save=False)

    if j == n_epochs - 1 and save_plot:
        dir_support([trained_plots_path])
        loss_plot(loss_list=loss_list, 
                  MC_error=MC_error, 
                  current_epoch=j, 
                  save=save_plot,
                  plot_path=full_plot_name)

print('Done! :)')

# We save the model
if save_model:
    dir_support([trained_models_path])
    state = {
        'epoch': n_epochs,
        'state_dict': vae.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'lr': lr,
        'N': N,
        's': latent_size,
        'hidden_size_enc': hidden_size_enc,
        'hidden_size_dec': hidden_size_dec,
        'MC_size': MC_size,
        'b': batch_size
        }
    torch.save(state, full_model_name)
    print(f'Model correctly saved at: {full_model_name}')
























