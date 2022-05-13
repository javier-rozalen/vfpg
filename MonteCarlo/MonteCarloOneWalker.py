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
from modules.plotters import S_vs_npaths,histo

################################# GENERAL PARAMETERS ##########################
seed = 5
N = 200 # 32 = 31 + 1
mu = 0
sigma = 1/6
M = 10000
leap = M/100
m = 1
w = 1
n_faulty = 60
T = 100

metropolis = True
erase_faulty_paths = False
add_symmetric_paths = False
plot_histogram = False
plot_type = 'symmetrized'

###############################################################################
np.random.seed(seed)
d = 1.
h = T/N

dict_of_plot = {'symmetrized':f'plots/one_walker/paths_N{N+1}_seed{seed}_symmetrized.png',
                'non-symmetrized':f'plots/one_walker/paths_N{N+1}_seed{seed}.png'}
filename = f'paths/one_walker/paths_N{N+1}_seed{seed}.txt'
filename_symmetrized = f'paths/one_walker/paths_N{N+1}_seed{seed}_symmetrized.txt'
file_to_plot = filename_symmetrized if plot_type == 'symmetrized' else filename
plot_name = dict_of_plot[plot_type]

x0 = np.random.normal(mu,sigma,N).tolist()
x0 = [0.]*N
paths = [x0]
S_paths = [S_HO(x0,h,m,w)]

################## METROPOLIS ####################
if metropolis:
    k = 0
    with open(filename,'w') as file:
        file.close()               
    with open(filename,'a') as file:
        pbar = tqdm(total=M)
        while len(paths)<M:
            chi = np.random.normal(mu,sigma,N)
            path_old = paths[-1]
            path_new = path_old+d*chi
            S_new = S_HO(path_new,h,m,w)
            S_old = S_HO(path_old,h,m,w)
            delta_S = S_new-S_old
            if delta_S<=0:
                # Path accepted
                good_path = path_new
                good_S = S_new
                accepted = True
            else:
                r = np.random.rand(1)
                if r<np.exp(-delta_S):
                    # Path accepted
                    good_path = path_new
                    good_S = S_new
                    accepted = True
                else:
                    # Path rejected
                    good_path = path_old
                    good_S = S_old
                    accepted = False
                    
            if accepted:
                paths.append(good_path)
                S_paths.append(good_S)
                file.write(' '.join([str(x) for x in good_path])+f' {str(good_path[0])}\n')
                pbar.update(1)
            
                if len(paths)%leap == 0 and len(paths)>60:
                    x_axis,x_axis_sym = [],[]
                    for path in paths[60:]:
                        x_axis.append(path[0])
                        x_axis_sym.append(path[0])
                        x_axis_sym.append(-path[0])
                    save_plot = True if len(paths)==M else False
                    name_of_plot = plot_name if len(paths)==M else ''
                    histo(x_axis,x_axis_sym,S_paths[60:],M,N,seed,name_of_plot,save_plot)
                    if save_plot:
                        print('Plot saved.')
              
            k += 1

        file.close()
        pbar.close()

        
################## ERASE FAULTY PATHS ####################
if erase_faulty_paths:        
    # We get rid of the first n_faulty paths
    list_of_good_paths = []
    with open(filename,'r') as file:
        c = 0
        for path in file.readlines():
            if c>n_faulty:
                list_of_good_paths.append(path)
            c += 1
        file.close()
    with open(filename,'w') as file:
        for good_path in list_of_good_paths:
            file.write(good_path)
        file.close()
    #os.rename(filename,filename_filtered)
    
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






    


