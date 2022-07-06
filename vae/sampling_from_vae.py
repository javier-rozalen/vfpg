# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 13:46:29 2022

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
import torch, math, time
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy import integrate

# My modules
from modules.neural_networks import VAE
from modules.plotters import loss_plot, histogram_plot, master_plot
from modules.aux_functions import S_HO, fetch_data, histogram
from modules.physical_constants import *
from modules.loss_functions import ELBO

######################## PARAMETERS ########################
# General parameters
dev = 'cpu'
n_paths = 5000
fixed_endpoints = False
trained_models_path = 'saved_models/'

# Plotting parameters
t_0 = 0.
t_f = 100
dx = 0.1 # histogram bin size
bound = 6
save_plot = False
plot_path = 'saved_plots/'
plot_name = 'histogram.pdf'
full_plot_path = plot_path + plot_name

if fixed_endpoints:
    trained_models_path += 'fixed_endpoints/'
else:
    trained_models_path += 'free_endpoints/'

######################## LOADING THE TRAINED MODEL ########################
# copy from the training console output: 
trained_model = trained_models_path + 'various_s/free_nepochs600_lr0.001_N20_n2000_b150_s3_resumedFalse_testTrue.pt' 

# Variatonal AutoEncoder
sample_size = torch.load(trained_model, map_location=torch.device(dev))['N']
batch_size = torch.load(trained_model, map_location=torch.device(dev))['b']
latent_size = torch.load(trained_model, map_location=torch.device(dev))['s']
MC_size = torch.load(trained_model, map_location=torch.device(dev))['MC_size']
hidden_size_enc = torch.load(trained_model, map_location=torch.device(dev))['hidden_size_enc']
hidden_size_dec = torch.load(trained_model, map_location=torch.device(dev))['hidden_size_dec']

vae = VAE(dev='cpu',
          sample_size=sample_size,
          batch_size=batch_size,
          latent_size=latent_size,
          MC_size=MC_size,
          hidden_size_enc=hidden_size_enc,
          hidden_size_dec=hidden_size_dec).to(dev)
vae.load_state_dict(torch.load(trained_model, map_location=torch.device(dev))['state_dict'])
vae.eval()

# Loss function (in case we want to resume the training)
loss_fn = ELBO

# Optimizer (in case we want to resume the training)
lr = torch.load(trained_model, map_location=torch.device(dev))['lr']
optimizer = torch.optim.Adam(params=vae.parameters(), lr=lr)
optimizer.load_state_dict(torch.load(trained_model, map_location=torch.device(dev))['optim_state_dict'])

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
print(f'Sampling {n_paths} time: {round(tf-t0, 3)} sec. on {dev}.')

######################## HISTOGRAM ########################
# We compute the density histogram from the paths and pass it to the plotter
dx_list = [dx for e in range(n_paths)]
counts = list(map(histogram, paths.cpu().detach().numpy(), dx_list))
wf2 = np.sum(counts, axis=0)
wf_norm = integrate.simpson(y=wf2, x=np.linspace(-4.95, 4.95, 100))
wf2 /= wf_norm

x_axis = np.linspace(-4.95, 4.95, 100)

if save_plot:
    dir_support([plot_path])
    
time_grid = [e for e in np.linspace(t_0, t_f, sample_size)]
histogram_plot(x_axis=x_axis,
               y_axis=wf2,
               path_manifold=paths,
               bound=bound,
               time_grid=time_grid,
               save_plot=save_plot,
               plot_path=full_plot_path)


