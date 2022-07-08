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

######################## SCRIPT PARAMETERS ########################
save_plot = True
plot_path = 'computation_time.pdf'

################ DATA FETCHING ################
MCMC_file = '../MonteCarlo/saved_data/computation_time/N20/times.txt'
VAE_file = '../vae/computation_time/N20/times.txt'

MCMC_offset = 0.0
VAE_offset = 170
n_paths = []
MCMC_times = []
VAE_times = []

with open(MCMC_file, 'r') as file:
    for line in file.readlines():
        time = float(line.split(' ')[1])
        MCMC_times.append(time)
    file.close()

with open(VAE_file, 'r') as file:
    for line in file.readlines():
        paths = int(line.split(' ')[0])
        time = float(line.split(' ')[1])
        n_paths.append(paths)
        VAE_times.append(time)
    file.close()

log_n_paths = [round(np.log10(n), 2) for n in n_paths]
log_MCMC_times = [round(np.log10(e + MCMC_offset), 2) for e in MCMC_times]
log_VAE_times = [round(np.log10(e + VAE_offset), 2) for e in VAE_times]

################ PLOTTING ################
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

ax.set_title('Computation time', fontsize=22)
ax.set_xlabel(r'$\log_{10} N_{paths}$', fontsize=20)
ax.set_ylabel(r'$\log_{10} t$', fontsize=20)
ax.tick_params(axis='both', labelsize=19)
#ax.ticklabel_format(useOffset=False)
ax.plot(log_n_paths, log_MCMC_times, label='MCMC')
ax.plot(log_n_paths, log_VAE_times, label='VAE', linestyle='--')
ax.legend(fontsize=18)

if save_plot:
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f'Plot correctly saved at: {plot_path}')
    
plt.show()







