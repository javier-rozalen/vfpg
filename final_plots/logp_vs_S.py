# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:56:04 2022

@author: javir
"""
#%% ####################### IMPORTS ########################
# Change to the directory of this script
import pathlib, os, sys
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('.')

# General modules
import numpy as np
from tqdm import tqdm
import torch, math
import matplotlib.pyplot as plt

# My modules
from modules.neural_networks import VAE
from modules.plotters import loss_plot
from modules.aux_functions import *
from modules.physical_constants import *
from modules.loss_functions import ELBO

#%% ####################### PARAMETERS ########################
dev = 'cpu'
n_paths = 500
trained_models_path = '../vae/saved_models/free_endpoints/'

######################## AUXILIARY FUNCTIONS ########################
toggle_prints = False

def print_(*message):
    if toggle_prints:
        print(*message)

######################## LOADING THE TRAINED MODEL ########################
# copy from the training console output: 
trained_model = trained_models_path + 'various_s/free_nepochs600_lr0.001_N20_n2000_b150_s3_resumedFalse_testTrue.pt' 

# Variatonal AutoEncoder
sample_size = torch.load(trained_model, map_location=torch.device(dev))['N']
batch_size = 1
latent_size = torch.load(trained_model, map_location=torch.device(dev))['s']
MC_size = n_estimate
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

######################## TEST SET ########################

# Sampling z from prior distribution
prior_z_loc = torch.zeros(n_paths, latent_size).to(dev)
prior_z_cov_mat = torch.eye(latent_size).reshape(1, latent_size, latent_size)
prior_z_cov_mat = prior_z_cov_mat.repeat(n_paths, 1, 1).to(dev)
prior_z_dist = MultivariateNormal(loc=prior_z_loc, 
                                  covariance_matrix=prior_z_cov_mat)

t0 = time.time()
zs = prior_z_dist.sample() # shape = [n_paths, latent_size]

# Sampling paths from zs
test_set = vae.decoder_sampler(zs) # shape [n_paths, sample_size]
tf = time.time()
print(f'Sampling {n_paths} paths time: {round(tf-t0, 3)} sec. on {dev}.')

#%% ############################## TRAIN SET ###############################
N = 20
M = 10000
M_bound = 2000
paths_file = f'../MonteCarlo/saved_data/paths_N{N}_M{M}.txt'
actions_file = f'../MonteCarlo/saved_data/actions_N{N}_M{M}.txt'
train_set, actions_set = fetch_data(M, paths_file, actions_file)
train_set = train_set[:train_set.size(0)-M_bound].to(dev)

#%% ######################### COMPUTING STUFF ############################
save_to_file = True
set_ = test_set
n_estimate = 100
if set_ == 'train_set':
    file_name = '../vae/data/master_plot_train_set.txt'  
else:    
    file_name = '../vae/data/master_plot_test_set.txt'

logs_prob = []
actions = []
with open(file_name, 'a') as file:
    for i in tqdm(range(set_.size(0))):
        path = set_[i].unsqueeze(0)
        expectation_list = []
        for j in range(n_estimate):
            z, enc_mus, enc_sigmas2, dec_mus, dec_sigmas2 = vae(path)
            
            # Prior of z is a Normal(0, 1)
            prior_z_loc = torch.zeros(1, latent_size).to(dev)
            prior_z_cov_mat = torch.eye(latent_size).reshape(1, latent_size, latent_size)
            prior_z_cov_mat = prior_z_cov_mat.repeat(1, 1, 1).to(dev)
            print_(f'prior_z loc: {prior_z_loc}', prior_z_loc.shape)
            print_(f'prior_z_cov_mat: {prior_z_cov_mat}', prior_z_cov_mat.shape)
            prior_z_dist = MultivariateNormal(loc=prior_z_loc, 
                                              covariance_matrix=prior_z_cov_mat)
            
            # Encoder prob is a Normal(enc_mus, enc_sigmas)
            enc_cov_mat = torch.diag_embed(enc_sigmas2)
            enc_dist = MultivariateNormal(loc=enc_mus, 
                                          covariance_matrix=enc_cov_mat)
            print_(f'enc_loc: {enc_mus}', enc_mus.shape)
            print_(f'enc_cov_mat: {enc_cov_mat}', enc_cov_mat.shape)
            
            # Decoder prob is a Normal(dec_mus, dec_sigmas)
            dec_cov_mat = torch.diag_embed(dec_sigmas2)
            dec_dist = MultivariateNormal(loc=dec_mus, 
                                          covariance_matrix=dec_cov_mat)
            print_(f'dec_loc: {dec_mus}', dec_mus.shape)
            print_(f'dec_cov_mat: {dec_cov_mat}', dec_cov_mat.shape)
        
            q_z_x = torch.exp(enc_dist.log_prob(z))
            p_z = torch.exp(prior_z_dist.log_prob(z))
            p_x_z = torch.exp(dec_dist.log_prob(path))
            expectation_term = (p_z * p_x_z / q_z_x).detach()
            expectation_list.append(expectation_term)
            
        action = S_HO(path.squeeze(), 100/20, m, w).detach()
        p_x = torch.sum(torch.stack(expectation_list)) / n_estimate
        log_p_x = torch.log(p_x)
        
        if save_to_file:
            file.write(f'{action.item()} {log_p_x.item()}\n')
        logs_prob.append(log_p_x)
        actions.append(action)
    file.close()

#%% ############################## PLOTTING ###############################


# Data fetching
actions_train = []
logs_prob_train = []
with open('../vae/data/master_plot_train_set.txt', 'r') as file:
    for line in file.readlines():
        a, l = float(line.split(' ')[0]), float(line.split(' ')[1])
        actions_train.append(a)
        logs_prob_train.append(l)
    file.close()
    
actions_test = []
logs_prob_test = []
with open('../vae/data/master_plot_test_set.txt', 'r') as file:
    for line in file.readlines():
        a, l = float(line.split(' ')[0]), float(line.split(' ')[1])
        actions_test.append(a)
        logs_prob_test.append(l)
    file.close()
    
n_accepted_MC = []
actions_MC = []
with open('aux_file.txt', 'r') as file:
    for line in file.readlines():
        n = int(line.split(' ')[0])
        a = float(line.split(' ')[1])
        n_accepted_MC.append(n)
        actions_MC.append(a)
    file.close()

def master_plot(x_axis_train, y_axis_train, x_axis_test, y_axis_test,
                save_plot=False, plot_path=''):
    
    """Plots logp(x) vs S(x)/hbar"""
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.25)
       
    # logp vs S
    ax = axes[0]
    ax.set_title(r'Path distribution', fontsize=17)
    ax.set_xlabel(r'$S_E(\mathbf{x})/\hbar$', fontsize=17)
    ax.set_ylabel(r'$\log p(\mathbf{x})$', fontsize=17)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_xlim(0, 20)
    ax.set_ylim(-20, 0)
    
    ax.scatter(x_axis_train, y_axis_train, color='blue', label='Train') # Train data
    ax.scatter(x_axis_test, y_axis_test, facecolors='none', edgecolors='orange', label='Test') # Test data
    ax.legend(fontsize=15)
    
    # MC
    ax = axes[1]
    ax.set_title(r'$S_E[x(\tau)]$', fontsize=17)
    ax.set_xlabel(r'$N_{paths}$', fontsize=17)
    ax.tick_params(axis='both', labelsize=15)
    ax.plot(n_accepted_MC[:2000], actions_MC[:2000])
    
    # Save
    if save_plot:
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        print(f'Plot correctly saved at: {plot_path}')
    
    plt.show()
    
# Plotting
save_plot = True
plot_path = 'vae_logp_vs_S.pdf'
master_plot(x_axis_train=actions_train,
            y_axis_train=logs_prob_train,
            x_axis_test=actions_test,
            y_axis_test=logs_prob_test,
            save_plot=save_plot,
            plot_path=plot_path)



