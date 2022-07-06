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
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
N = 20
M = 10000
M_bound = 2000
fixed_endpoints = False
x0 = torch.tensor(0.5) # fixed endpoint 
testing = True

# Names of files, directories
paths_file = f'../MonteCarlo/saved_data/paths_N{N}_M{M}.txt'
actions_file = f'../MonteCarlo/saved_data/actions_N{N}_M{M}.txt'
trained_models_path = 'saved_models/'
trained_plots_path = 'saved_plots/'
subdirectory = 'tests/'

# Neural network parameters
latent_size = 3
hidden_size_enc = 15
hidden_size_dec = 15    

# Training parameters
n_epochs = 30
batch_size = 150
MC_size = 1000
lr = 1e-3

# Plotting parameters
n_plots = 100
adaptive_factor = 2.5
leap = n_epochs/n_plots

# Saves/Booleans
save_model = True
save_plot = True
show_periodic_plots = True
continue_from_last = False

if fixed_endpoints:
    endpoint = round(x0.item(), 2)
    trained_model_name = (f'fixed{endpoint}_nepochs{n_epochs}_lr{lr}_N{N}'
                          f'_n{MC_size}_b{batch_size}_s{latent_size}_'
                          f'resumed{continue_from_last}_test{testing}')
    trained_models_path += 'fixed_endpoints/' + subdirectory
    trained_plots_path += 'fixed_endpoints/' + subdirectory

else:
    trained_model_name = (f'free_nepochs{n_epochs}_lr{lr}_N{N}'
                          f'_n{MC_size}_b{batch_size}_s{latent_size}_'
                          f'resumed{continue_from_last}_test{testing}')
    trained_models_path += 'free_endpoints/' + subdirectory
    trained_plots_path += 'free_endpoints/' + subdirectory

full_model_name = trained_models_path + trained_model_name + '.pt'
full_plot_name = trained_plots_path + trained_model_name + '.pdf'

# copy here the path of the model to resume training:
model_to_resume = trained_models_path + 'free_nepochs600_lr0.001_N20_n2000_b150_s1_resumedFalse_testFalse.pt'

######################## DATA FETCHING ########################
print('Fetching data...')
train_set, actions_set = fetch_data(M, paths_file, actions_file)
if fixed_endpoints:
    idcs = []
    for i in range(len(train_set)):
        if abs(train_set[i][0]-x0) <= 1e-1:
            idcs.append(i)
    idcs = torch.tensor(idcs)     
    M_bound = len(idcs)  
    batch_size = 50
    train_set = torch.index_select(train_set, 0, idcs).to(dev)
    print(f'train_set: {train_set}', train_set.shape)
else:
    # The last M_bound paths are (hopefully) better distributed
    train_set = train_set[:train_set.size(0)-M_bound].to(dev)
print('Data fetching complete.\n')

# train_set shape: [M_bound, N]
######################## NEURAL NETWORK ########################
loss_fn = ELBO

if continue_from_last:
    print('Resuming training...')
    sample_size = torch.load(model_to_resume)['N']
    batch_size = torch.load(model_to_resume)['b']
    latent_size = torch.load(model_to_resume)['s']
    MC_size = torch.load(model_to_resume)['MC_size']
    hidden_size_enc = torch.load(model_to_resume)['hidden_size_enc']
    hidden_size_dec = torch.load(model_to_resume)['hidden_size_dec']
    vae = VAE(dev=dev,
              sample_size=N,
              batch_size=batch_size,
              latent_size=latent_size,
              MC_size=MC_size,
              hidden_size_enc=hidden_size_enc,
              hidden_size_dec=hidden_size_dec).to(dev)
    vae.load_state_dict(torch.load(model_to_resume)['state_dict'])
    lr = torch.load(model_to_resume)['lr']
    optimizer = torch.optim.Adam(params=vae.parameters(), lr=lr)
    optimizer.load_state_dict(torch.load(model_to_resume)['optim_state_dict'])
else:
    vae = VAE(dev=dev,
              sample_size=N,
              batch_size=batch_size,
              latent_size=latent_size,
              MC_size=MC_size,
              hidden_size_enc=hidden_size_enc,
              hidden_size_dec=hidden_size_dec).to(dev)
    optimizer = torch.optim.Adam(params=vae.parameters(), lr=lr)

######################## TRAINING LOOP #########################
print(f'Training a model with {count_params(vae)} parameters on {dev}.')
print('---------------------------------------------------------------')
print(vae)
print('---------------------------------------------------------------\n')
print(f'Training set size: {M_bound}')
print(f'Example paths endpoints: {"fixed" if fixed_endpoints else "free"}')
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
    for b in range(M_bound // batch_size):
        #print(f'batch {b}')
        z_b = train_set[b*batch_size:(b+1)*batch_size, :]
        loss, MC_error = train_loop(dev=dev,
                                    model=vae,
                                    loss_fn=loss_fn,
                                    optimizer=optimizer,
                                    train_set=z_b)
        #print(loss)
    loss_list.append(loss)
    
    if ((j+1) % leap) == 0 and show_periodic_plots:
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
























