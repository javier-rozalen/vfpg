#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:01:52 2022

@author: jozalen
"""

################################# IMPORTS #####################################
# General
import numpy as np
import random,os
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Manager,Pool,freeze_support

# My Modules
from modules.actions import S_HO
from modules.plotters import S_vs_npaths

################################# GENERAL PARAMETERS ##########################
seed = 3
metropolis = True
erase_faulty_paths = False
symmetric_paths = False
plot_histo = False
N = 49 # 32 = 31 + 1
mu = 0
sigma = 1/6
M = 10000
leap = 10000
n_faulty = 0
m = 1
w = 1

###############################################################################
random.seed(seed)
paths = np.array([np.random.normal(mu,sigma,N) for _ in range(M)])
#paths = np.array([[1 for _ in range(N)] for _ in range(M)])
h = 1/N

filename = f'paths/paths_seed{seed}_N{N+1}.txt'
filename_filtered = f'paths/paths_seed{seed}_N{N+1}_filtered.txt'
filename_filtered_symmetrized = f'paths/paths_seed{seed}_N{N+1}_filtered_symmetrized.txt'

def probs_and_stuff(c,y,paths,paths_new,accepted_paths,S_paths):
    delta_S = S_HO(y[c],h,m,w)-S_HO(paths[c],h,m,w)
    if delta_S<0:
        # Path accepted
        paths_new[c] = y[c]
        #file.write(' '.join([str(x) for x in y[c].tolist()])+f' {str(y[0])}\n')
        accepted_paths.append(y[c])
        S_paths.append(S_HO(y[c],h,m,w))
    else:
        r = np.random.rand(1)
        if r<np.exp(-delta_S):
            # Path accepted
            paths_new[c] = y[c]
            #file.write(' '.join([str(x) for x in y[c].tolist()])+f' {str(y[0])}\n')
            accepted_paths.append(y[c])
            S_paths.append(S_HO(y[c],h,m,w))

################## METROPOLIS ####################
if metropolis:
    k = 0
    if __name__ == '__main__':
        manager = Manager()
        accepted_paths = manager.list()
        S_paths = manager.list()
        with open(filename,'w') as file:
            file.close()
        with open(filename,'a') as file:
            pbar = tqdm(total=M)
            while len(accepted_paths)<M:
                chis = np.array([np.random.normal(mu,sigma,N) for _ in range(M)])
                y = paths+chis
                paths_new = paths
                
                pool = Pool(processes=4)
                pool.starmap(probs_and_stuff,[(c,y,paths,paths_new,accepted_paths,S_paths) for c in range(M)])
                pool.close()
                pool.join()
                paths = paths_new
                    
                if (k+1)%leap == 0:
                    #S_vs_npaths(paths,S_paths)
                    paths
                    
                k += 1
                pbar.update(1)
                print('hey or something')
                #print(f' Acceptance : {acc}')
                #print(f'# accepted paths : {len(accepted_paths)}')
            pbar.close()
            file.close()
            
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
    os.rename(filename,filename_filtered)
    
################## ADD THE SYMMETRIC PATHS ####################
if symmetric_paths: 
    with open(filename_filtered_symmetrized,'w') as paths_symm_file:
        paths_symm_file.close()
    with open(filename_filtered,'r') as paths_file:
        with open(filename_filtered_symmetrized,'a') as paths_symm_file:
            for path in paths_file.readlines():
                symmetric_path = [-float(x) for x in path.split(' ')]
                symmetric_path = ' '.join([str(x) for x in symmetric_path])
                paths_symm_file.write(path)
                paths_symm_file.write(symmetric_path+' \n')
            paths_symm_file.close()
        paths_file.close()

################## WAVE FUNCTION COMPUTATION ####################
if plot_histo:
    x_axis = []
    with open(filename_filtered_symmetrized,'r') as paths_file:
        for path in paths_file.readlines():
            x0 = float(path.split(' ')[0])
            x_axis.append(x0)
            #path_prime = [float(x) for x in path.split(' ')]
        paths_file.close()
                
    plt.hist(x_axis, density=True, bins=50)  # density=False would make counts
    plt.ylabel('$|\Psi(x)|^2$')
    plt.xlabel('x')
    plt.title(f'M = {M}, N = {N+1}, Seed = {seed}')

    


