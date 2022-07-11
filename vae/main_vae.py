# -*- coding: utf-8 -*-
####################### IMPORTANT INFORMATION #######################
"""
This program trains a Variational AutoEncoder (VAE) to solve the quantum harmonic
oscillator (HO) 1D. The VAE learns features of a given set of 1D particle paths 
generated via Markov Chain Monte Carlo (MCMC) and then is able to generate new
paths which follow, ideally, a distribution based on their action. The models 
are saved under the saved_models/ folder. 
Parameters to adjust manually:
    GENERAL PARAMETERS
    N --> int, Number of points of each path generated with MCMC.
    M --> int, Number of paths generated with MCMC.
    M_bound --> int, Number of paths (out of M) that will actually be used to
                train the VAE.
    fixed_endpoints --> Boolean, Whether we want to train the VAE only on paths
                        with equal endpoints.
    x0 --> tensor, The specific endpoint of the paths (only applies if 
            if fixed_endpoints is set to True).
    testing --> Boolean, Whether the actual run is a test or not. If True, the
                trained model and plot will include this information in the
                file name.
    
    FILE PARAMETERS
    paths_file --> str, Path to the MCMC-generated paths.
    actions_file = str, Path to the actions of the MCMC-generated paths.
    trained_models_path = --> str, Path where the trained model will be saved.
    trained_plots_path = --> str, Path where the plot of the trained model will
                            be saved.
    subdirectory = --> str, Subdirectory under the saved_models/ folder where
                        the trained model will be saved.
    resumed_name --> str, Name of the file of the previously trained model. This
                    only has effect if continue_from_last is set to True

    NEURAL NETWORK PARAMETERS
    latent_size --> int, Dimension of the latent space.
    hidden_size_enc --> int, Number of hidden nodes for the encoder.
    hidden_size_dec --> int, Number of hidden nodes for the decoder.

    TRAINING PARAMETERS
    n_epochs --> int, Number of training epochs.
    batch_size --> int, Batch size (Stochastic Gradient Descent). 
    MC_size --> int, Number of latent variables used to estimate the expectation
                that appears in the ELBO loss.
    lr --> float, Learning rate.

    PLOTTING PARAMETERS
    n_plots --> int, Total number of plots that are generated during training.

    SAVES/BOOLEANS
    save_model --> Boolean, Whether the trained model is saved at the end.
    save_plot --> Boolean, Whether the plot of the trained model is saved at
                    the end.
    show_periodic_plots = Boolean, Whether plots are periodically shown during
                        training.
    continue_from_last = Boolean, Whether the training is resumed from a 
                        previously trained model.
                        
"""

######################## IMPORTS ########################
# Change to the directory of this script
import pathlib, os, sys
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('.')

# General modules
from tqdm import tqdm
import torch

# Our modules
from modules.neural_networks import VAE
from modules.plotters import loss_plot
from modules.aux_functions import *
from modules.physical_constants import *
from modules.loss_functions import ELBO

######################## GENERAL PARAMETERS ########################
# General parameters
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
N = 20
M = 10000
M_bound = 2000
fixed_endpoints = False
x0 = torch.tensor(0.5) 
testing = True

# File parameters
paths_file = f'../MonteCarlo/saved_data/paths/paths_N{N}_M{M}.txt'
actions_file = f'../MonteCarlo/saved_data/actions/actions_N{N}_M{M}.txt'
trained_models_path = 'saved_data/models/'
trained_plots_path = 'saved_data/plots/'
subdirectory = ''
resumed_name = ''

# Neural network parameters
latent_size = 3
hidden_size_enc = 100
hidden_size_dec = 100  

# Training parameters
n_epochs = 600
batch_size = 150
MC_size = 2000
lr = 1e-3

# Plotting parameters
n_plots = 10
leap = n_epochs/n_plots

# Saves/Booleans
save_model = True
save_plot = True
show_periodic_plots = True
continue_from_last = False

# Not adjustable
if fixed_endpoints:
    endpoint = round(x0.item(), 2)
    trained_model_name = (f'fixed{endpoint}_nepochs{n_epochs}_lr{lr}_N{N}'
                          f'_n{MC_size}_b{batch_size}_s{latent_size}_h{hidden_size_enc}_'
                          f'Mbound{M_bound}_resumed{continue_from_last}_test{testing}')
    trained_models_path += 'fixed_endpoints/' + subdirectory
    trained_plots_path += 'fixed_endpoints/' + subdirectory

else:
    trained_model_name = (f'free_nepochs{n_epochs}_lr{lr}_N{N}'
                          f'_n{MC_size}_b{batch_size}_s{latent_size}_h{hidden_size_enc}_'
                          f'Mbound{M_bound}_resumed{continue_from_last}_test{testing}')
    trained_models_path += 'free_endpoints/' + subdirectory
    trained_plots_path += 'free_endpoints/' + subdirectory

full_model_name = trained_models_path + trained_model_name + '.pt'
full_plot_name = trained_plots_path + trained_model_name + '.pdf'
model_to_resume = trained_models_path + resumed_name

######################## DATA FETCHING ########################
# Here, the MCMC-generated paths and their actions are loaded to memory to
# be used as the training set.
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
    train_set = train_set[:train_set.size(0)-M_bound].to(dev)
print('Data fetching complete.\n')

# train_set shape: [M_bound, N]
######################## NEURAL NETWORK ########################
# Here we load the VAE to memory.

loss_fn = ELBO # Loss function to use

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
# Here write the training (epoch) loop with mini-batching. 
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
        z_b = train_set[b*batch_size:(b+1)*batch_size, :]
        loss, MC_error = train_loop(dev=dev,
                                    model=vae,
                                    loss_fn=loss_fn,
                                    optimizer=optimizer,
                                    train_set=z_b)
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
    print(f'Model correctly saved at: {full_model_name}.')
























