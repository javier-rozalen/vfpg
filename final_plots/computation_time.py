# -*- coding: utf-8 -*-

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
import matplotlib.pyplot as plt

# My modules
from modules.neural_networks import VAE
from modules.plotters import loss_plot
from modules.aux_functions import *
from modules.physical_constants import *
from modules.loss_functions import ELBO

######################## SCRIPT PARAMETERS ########################
save_plot = False
plot_path = ''

################ DATA FETCHING ################
MCMC_file = 'MonteCarlo/saved_data/computation_time/N20/times.txt'
VAE_file = 'vae/computation_time/N20/times.txt'

n_paths = [500, 
           1000, 2000, 3000, 4000, 5000,
           10000, 20000, 30000, 40000, 50000,
           100000, 200000, 300000, 400000, 500000, 
           1000000, 2000000, 3000000, 4000000, 5000000]
MCMC_offset = 0.5
VAE_offset = 30*60
MCMC_times = []
VAE_times = []

with open(MCMC_file, 'r') as file:
    for line in file.readlines():
        time = float(line.split(' ')[1])
        MCMC_times.append(time)
    file.close()
    
with open(VAE_file, 'r') as file:
    for line in file.readlines():
        time = float(line.split(' ')[1])
        VAE_times.append(time)
    file.close()
    
################ PLOTTING ################
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

ax.set_title('Computation time')
ax.set_xlabel(r'$\log N_{paths}$', fontsize=17)
ax.set_ylabel(r'$\log t$', fontsize=17)
ax.tick_params(axis='both', labelsize=15)
ax.plot(n_paths, MCMC_times, label='MCMC')
ax.plot(n_paths, VAE_times, label='VAE')
ax.legend(fontsize=16)

if save_plot:
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')

plt.show()







