#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:01:52 2022

@author: jozalen
"""
import pathlib, os, sys
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('.')
################################# IMPORTS #####################################
# General
import numpy as np
from tqdm import tqdm
import scipy.integrate as integrate
import time

# My Modules
from modules.actions import S_HO, S_double_well
from modules.plotters import histo2
from modules.dir_support import dir_support

################################# GENERAL PARAMETERS ##########################
numbers_of_paths = [710]
seed = 1
N = 20
mu = 0
sigma = 1/6

m = 1
w = 1
n_faulty = 300
T = 100
d = 1.
dx = 0.1
hbar = 1.
action = S_HO
metropolis = True
write_data = False

np.random.seed(seed)
h = T/N

dir_support(['saved_data', 'computation_time', f'N{N}'])
file = f'saved_data/computation_time/N{N}/times.txt'
x_axis = np.linspace(-4.95, 4.95, 100)

def histograma(x, dx):
    """
    Counts the frequency of appearence of points in a 100-point grid.

    Parameters
    ----------
    x : list/numpy array
        Path.
    dx : float
        Grid scale.

    Returns
    -------
    numpy array
        1D position grid with N=100, dx=dx.

    """
    count = [0]*100
    n = len(x)
	
    for i in range(n):
		
        j = 0
        done = False
        while -5 + j*dx <= +5 and done == False:
			
            if x[i] >= -5 + j*dx and x[i] <= -5 + (j + 1)*dx:
                try:
                    count[j] += 1
                except:
                    pass
                done = True
            else:
                j += 1
    return np.array(count)

with open(file, 'a') as file:
    for M in numbers_of_paths:
        x0 = [0.]*N
        #x0 = np.random.normal(0, 1, N).tolist()
        paths = [x0]
        S_paths = [action(x0, h, m, w)]
        wf = np.array([0.]*100)
        leap = M
        ################## METROPOLIS ####################
        first_accepted = 0
        t0 = time.time()
        if metropolis:
            k = 0
            n_accepted = 1
            pbar = tqdm(total=M)
            while n_accepted < M:
                chi = np.random.normal(mu, sigma, N)
                path_old = paths[-1]
                path_new = path_old + d*chi
                path_new[-1] = path_new[0]
                S_new = action(path_new, h, m, w)
                S_old = action(path_old, h, m, w)
                delta_S = S_new - S_old
                
                if delta_S <= 0:
                    accepted = True
                else:
                    r = np.random.rand(1)
                    if r<np.exp(-delta_S):
                        accepted = True
                    else:
                        accepted = False
                  
                if accepted:
                    n_accepted += 1
                    paths.append(path_new)
                    S_paths.append(S_new)
                    pbar.update(1)
                    if n_accepted == n_faulty:
                        tb = time.time()
                    if n_accepted > n_faulty:
                        wf += histograma(path_new, dx) + histograma(-path_new, dx)
                        
                        if n_accepted % leap == 0:
                            wf_norm = integrate.simpson(y=wf, x=x_axis)
                            histo2(x_axis,wf/wf_norm, S_paths, n_accepted,
                                   path_new)
                        if n_accepted == M-1:
                            wf_norm = integrate.simpson(y=wf, x=x_axis)
                            histo2(x_axis, wf/wf_norm, S_paths, n_accepted,
                                   path_new, 'mc.pdf', save=False)
              
                k += 1
            pbar.close()
                
        tf = time.time()
        total_time = tf-t0
        #burn_in_time = tb-t0
        print(f'npaths = {M} complete.\n')
        if write_data:
            file.write(f'{M} {total_time}\n')
        
file.close()
    