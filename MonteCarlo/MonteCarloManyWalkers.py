#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:01:52 2022

@author: jozalen
"""

################################# IMPORTS #####################################
# General
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# My Modules
from modules.actions import S_HO
from modules.plotters import histo

################################# GENERAL PARAMETERS ##########################
seed = 1
n_steps = 500
leap = 5
N = 100 # 32 = 31 + 1
mu = 0
sigma = 1/6
M = 10000
m = 1
w = 1
hbar = 1
T = 10

metropolis = True
add_symmetric_paths = True
plot_histogram = True
save_plot = True
plot_type = 'symmetrized'

###############################################################################
np.random.seed(seed)
d = 10.
h = T/N
paths = np.array([np.random.normal(mu,sigma,N) for _ in range(M)])
S_paths = np.array([S_HO(x,h,m,w) for x in paths])

dict_of_plot = {'symmetrized':f'plots/many_walkers/paths_N{N+1}_steps{n_steps}_seed{seed}_symmetrized.png',
                'non-symmetrized':f'plots/many_walkers/paths_N{N+1}_steps{n_steps}_seed{seed}.png'}
filename = f'paths/many_walkers/paths_N{N+1}_steps{n_steps}_seed{seed}.txt'
filename_symmetrized = f'paths/many_walkers/paths_N{N+1}_steps{n_steps}_seed{seed}_symmetrized.txt'
file_to_plot = filename_symmetrized if plot_type == 'symmetrized' else filename
plot_name = dict_of_plot[plot_type]

S_paths = []

################## METROPOLIS ####################
if metropolis:
    if not os.path.exists(filename):                   
        for step in tqdm(range(n_steps)):
            acc = 0
            chis = np.array([np.random.normal(mu,sigma,N) for _ in range(M)])
            y = paths+d*chis
            for c in range(M):
                path_new = y[c]
                path_old = paths[c]
                S_new = S_HO(y[c],h,m,w)
                S_old = S_HO(paths[c],h,m,w)
                delta_S = S_new-S_old
                if delta_S<0:
                    # Path accepted
                    good_path = path_new
                    good_S = S_new
                    acc += 1
                else:
                    r = np.random.rand(1)
                    if r<np.exp(-delta_S):
                        # Path accepted
                         good_path = path_new
                         good_S = S_new
                         acc += 1
                    else:
                        # Path rejected
                        good_path = path_old
                        good_S = S_old
                        
                paths[c] = good_path
            
            if (step+1)%leap == 0:
                x_axis = []
                for path in paths:
                    x_axis.append(path[0])
                    x_axis.append(-path[0])
                histo(x_axis,M,N,seed)
                
        with open(filename,'w') as file:
            for path in paths:
                path_formatted = ' '.join([str(x) for x in path.tolist()])+f' {str(path.tolist()[0])}\n'
                file.write(path_formatted)
            file.close()
    else:
        print(f'Skipping already created file {filename}...')
    
################## ADD THE SYMMETRIC PATHS ####################
if add_symmetric_paths:
    if not os.path.exists(filename_symmetrized):
        with open(filename_symmetrized,'w') as paths_symm_file:
            paths_symm_file.close()
        with open(filename,'r') as paths_file:
            with open(filename_symmetrized,'a') as paths_symm_file:
                for path in paths_file.readlines():
                    symmetric_path = [-float(x) for x in path.split(' ')]
                    symmetric_path = ' '.join([str(x) for x in symmetric_path])
                    paths_symm_file.write(path)
                    paths_symm_file.write(symmetric_path+' \n')
                paths_symm_file.close()
            paths_file.close()
        print('Symmetric paths successfully added to file.')
    else:
        print('Skipping already symmetrized file...')

################## WAVE FUNCTION COMPUTATION ####################
if plot_histogram: 
    if not os.path.exists(plot_name):
        x_axis = []
        with open(file_to_plot,'r') as paths_file:
            for path in tqdm(paths_file.readlines()):
                x0 = float(path.split(' ')[0])
                x_axis.append(x0)
            paths_file.close()
                    
        plt.hist(x_axis, density=True, bins=50)  # density=False would make counts
        x_target = np.linspace(-6,6,200)
        y_target = (((1/np.pi)**(1/4))*np.exp(-x_target**2/2))**2
        plt.plot(x_target,y_target,label='$|\Psi_0(x)|^2$')
        plt.ylabel('$|\Psi(x)|^2$')
        plt.xlabel('x')
        plt.xlim(-3,3)
        plt.title(f'M = {M}, N = {N+1}, Seed = {seed}, {plot_type}')
        plt.legend()
        if save_plot:
            plt.savefig(plot_name)
            print(f'Plot saved {plot_name}.')
    else:
        print(f'Skipping the creation of already saved plot {plot_name}...')



