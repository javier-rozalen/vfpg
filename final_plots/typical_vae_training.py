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
import matplotlib.pyplot as plt

# My modules
from modules.neural_networks import VAE
from modules.aux_functions import histogram, dir_support
from modules.loss_functions import ELBO

######################## PARAMETERS ########################
# General parameters
dev = 'cpu'
n_paths = 10000
fixed_endpoints = False
trained_models_path = '../vae/saved_models/'

# Plotting parameters
t_0 = 0.
t_f = 100
dx = 0.1 # histogram bin size
bound = 6
save_plot = False
plot_path = './'
plot_name = 'typical_vae_training.pdf'
full_plot_path = plot_path + plot_name

if fixed_endpoints:
    trained_models_path += 'fixed_endpoints/'
else:
    trained_models_path += 'free_endpoints/'

######################## LOADING THE TRAINED MODEL ########################
# copy from the training console output: 
particular_model = 'various_M_bound/free_nepochs600_lr0.001_N20_n2000_b50_s3_h15_Mbound100_resumedFalse_testTrue.pt' 
trained_model = trained_models_path + particular_model
print(f'Testing model: {particular_model}')

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
def kltest(x, p, q):
    logp = np.array([np.log(e) if e!= 0. else 0. for e in p])
    logq = np.array([np.log(e) if e!= 0. else logp[list(q).index(e)] for e in q])
    
    kl = integrate.simpson(p * (logp - logq), x)
    
    return kl

def tfm_plot(x_axis, y_axis, y_MCMC, loss_list, n_epochs,
                   save_plot=False, plot_path=''):
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    
    # Wave function
    ax = axes[0]
    sigma = np.sqrt(1.)
    #x_target = np.linspace(-3, 3, 200)
    #y_target = (((1/(np.pi*sigma**2))**(1/4))*np.exp(-x_target**2/(2*sigma**2)))**2
    ax.hist(x_axis, weights=y_axis, bins=int(len(x_axis)))
    #ax.plot(x_axis,y_axis,linestyle='none',marker='o')
    ax.plot(x_axis, y_MCMC, label='MCMC fit')
    ax.set_xlim(-3, 3)
    #ax.set_ylim(0, 0.6)
    ax.set_ylabel('$|\Psi(x)|^2$', fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(fontsize=16)
    
    # Loss
    ax = axes[1]
    ax.set_title('Loss', fontsize=17)
    ax.set_xlabel('epochs', fontsize=17)
    ax.tick_params(axis='both', labelsize=15)
    x = [n for n in range(n_epochs)]
    ax.plot(x, loss_list)
    
    # Save
    if save_plot:
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        print(f'Plot correctly saved at: {plot_path}')
    
    plt.show()

# We compute the density histogram from the paths and pass it to the plotter
dx_list = [dx for e in range(n_paths)]
counts = list(map(histogram, paths.cpu().detach().numpy(), dx_list))
wf2 = np.sum(counts, axis=0)
wf_norm = integrate.simpson(y=wf2, x=np.linspace(-4.95, 4.95, 100))
wf2 /= wf_norm

x_axis = []
wf_MCMC = []
with open('../MonteCarlo/saved_data/wf_N20_M10000.txt', 'r') as file:
    for line in file.readlines():
        x = float(line.split(' ')[0])
        wf = float(line.split(' ')[1])
        x_axis.append(x)
        wf_MCMC.append(wf)

loss_list = []
with open('../vae/data/loss.txt', 'r') as file:
    for line in file.readlines():
        loss = float(line)
        loss_list.append(loss)
    file.close()

if save_plot:
    dir_support([plot_path])
    
#time_grid = [e for e in np.linspace(t_0, t_f, sample_size)]

tfm_plot(x_axis=x_axis,
         y_axis=wf2,
         y_MCMC=wf_MCMC,
         loss_list=loss_list,
         n_epochs = 600,
         save_plot=save_plot,
         plot_path=full_plot_path)

print(f'KL-div: {kltest(x_axis, np.array(wf_MCMC), wf2)}')

