# -*- coding: utf-8 -*-
####################### IMPORTANT INFORMATION #######################
"""
This program plots the results of the training of the VAE of file main_vae.py
It loads the trained VAE to memory and it samples a given number of paths from
it. From these paths, the GS density is estimated and plotted against the MCMC
reference density, and a subset of the generated paths is also plotted. 
Parameters to adjust manually:
    GENERAL PARAMETERS
    dev = 'cpu', 'cuda' --> Device in which to load the model.
    n_paths --> int, Number of paths to be sampled from the VAE. 
    fixed_endpoints --> Boolean, Whether to use models trained on paths with
                        fixed endpoints.
    trained_models_path --> str, Directory in which to search for trained models.
    concrete_model --> str, Name of the file that contains the trained model.

    PLOTTING PARAMETERS
    bound --> int, Number of paths which are plotted.
    save_plot --> Boolean, Whether to save the plot or not.
    plot_path --> str, Path where the plot will be saved. 
    plot_name --> str, Name with which the plot will be saved.
                        
"""

######################## IMPORTS ########################
# Change to the directory of this script
import pathlib, os, sys
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('.')

# General modules
import numpy as np
import torch, time
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy import integrate

# My modules
from modules.neural_networks import VAE
from modules.plotters import histogram_plot
from modules.aux_functions import histogram, dir_support
from modules.loss_functions import ELBO

######################## GENERAL PARAMETERS ########################
# General parameters
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
n_paths = 10000
fixed_endpoints = False
trained_models_path = 'saved_data/models/'
concrete_model = 'free_nepochs600_lr0.001_N20_n2000_b150_s3_h100_Mbound2000_resumedFalse_testTrue.pt' 

# Plotting parameters
bound = 4
save_plot = True
plot_path = 'saved_data/post_training_plots/'
plot_name = 'histogram.pdf'

# Not adjustable
full_plot_path = plot_path + plot_name
if fixed_endpoints:
    trained_models_path += 'fixed_endpoints/'
else:
    trained_models_path += 'free_endpoints/'

######################## LOADING THE TRAINED MODEL ########################
trained_model = trained_models_path + concrete_model

# Variatonal AutoEncoder
sample_size = torch.load(trained_model, 
                         map_location=torch.device(dev))['N']
batch_size = torch.load(trained_model, 
                        map_location=torch.device(dev))['b']
latent_size = torch.load(trained_model, 
                         map_location=torch.device(dev))['s']
MC_size = torch.load(trained_model, 
                     map_location=torch.device(dev))['MC_size']
hidden_size_enc = torch.load(trained_model, 
                             map_location=torch.device(dev))['hidden_size_enc']
hidden_size_dec = torch.load(trained_model, 
                             map_location=torch.device(dev))['hidden_size_dec']

vae = VAE(dev='cpu',
          sample_size=sample_size,
          batch_size=batch_size,
          latent_size=latent_size,
          MC_size=MC_size,
          hidden_size_enc=hidden_size_enc,
          hidden_size_dec=hidden_size_dec).to(dev)
vae.load_state_dict(torch.load(trained_model, 
                               map_location=torch.device(dev))['state_dict'])
vae.eval()

# Loss function (in case we want to resume the training)
loss_fn = ELBO

# Optimizer (in case we want to resume the training)
lr = torch.load(trained_model, map_location=torch.device(dev))['lr']
optimizer = torch.optim.Adam(params=vae.parameters(), lr=lr)
optimizer.load_state_dict(torch.load(trained_model, 
                                     map_location=torch.device(dev))['optim_state_dict'])

######################## PATH SAMPLING ########################

# Sampling z from prior distribution
prior_z_loc = torch.zeros(n_paths, latent_size).to(dev)
prior_z_cov_mat = torch.eye(latent_size).reshape(1, latent_size, latent_size)
prior_z_cov_mat = prior_z_cov_mat.repeat(n_paths, 1, 1).to(dev)
prior_z_dist = MultivariateNormal(loc=prior_z_loc, 
                                  covariance_matrix=prior_z_cov_mat)

t0 = time.time()
zs = prior_z_dist.sample() # shape = [n_paths, latent_size]

# Sampling paths from zs
paths = vae.decoder_sampler(zs) # shape [n_paths, sample_size]
tf = time.time()
print(f'Sampling {n_paths} paths time: {round(tf-t0, 3)} sec. on {dev}.')

######################## HISTOGRAM ########################
# We compute the density histogram from the paths and pass it to the plotter
dx = 0.1 
dx_list = [dx for e in range(n_paths)]
counts = list(map(histogram, paths.cpu().detach().numpy(), dx_list))
wf2 = np.sum(counts, axis=0)
wf_norm = integrate.simpson(y=wf2, x=np.linspace(-4.95, 4.95, 100))
wf2 /= wf_norm

x_axis = []
wf_MCMC = []
with open('../MonteCarlo/saved_data/wave_functions/wf_N20_M10000.txt', 'r') as file:
    for line in file.readlines():
        x = float(line.split(' ')[0])
        wf = float(line.split(' ')[1])
        x_axis.append(x)
        wf_MCMC.append(wf)

if save_plot:
    dir_support([plot_path])
    
t_0 = 0.
t_f = 100
time_grid = [e for e in np.linspace(t_0, t_f, sample_size)]

histogram_plot(x_axis=x_axis,
               y_axis=wf2,
               y_MCMC=wf_MCMC,
               path_manifold=paths,
               bound=10,
               time_grid=time_grid,
               save_plot=save_plot,
               plot_path=full_plot_path)


