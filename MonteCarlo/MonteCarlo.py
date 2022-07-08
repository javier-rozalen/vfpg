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
import scipy, math
import scipy.integrate as integrate

# My Modules
from modules.actions import S_HO, S_double_well
from modules.plotters import histo2, nice_plot
from modules.dir_support import dir_support

################################# GENERAL PARAMETERS ##########################
seed = 1
N = 20
mu = 0
sigma = 1/6
M = 10000
leap = M/1
m = 1
w = 1
n_faulty = 300
T = 100
d = 1.
dx = 0.1
hbar = 1.
action = S_HO
metropolis = True
save_wf = True
write_data = False
save_plot = False
x_axis = np.linspace(-4.95, 4.95, 100)

y_target = (((1/(np.pi*1**2))**(1/4))*np.exp(-x_axis**2/(2*1**2)))**2

if action == S_HO:
    paths_file = f'saved_data/paths_N{N}_M{M}.txt'
    actions_file = f'saved_data/actions_N{N}_M{M}.txt'
elif action == S_double_well:
    paths_file = f'saved_data/double_well/paths_N{N}_M{M}.txt'
    actions_file = f'saved_data/double_well/actions_N{N}_M{M}.txt'
###############################################################################
dir_support(['saved_data'])
np.random.seed(seed)
h = T/N

x0 = [0.]*N
x0 = np.random.normal(0, 1, N).tolist()
paths = [x0]
S_paths = [action(x0, h, m, w)]
wf = np.array([0.]*100)

def histograma(x,dx):
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

def kstest(x):
    global Fx_ref, F_n
    Fx_ref = np.array([0.94*np.sqrt(e**2)*math.erf(0.707*np.sqrt(e**2))/e + 0.94 for e in x])
    F_n = np.cumsum(x) * dx
    
    return max(abs(F_n - Fx_ref))
    
def kltest(x, p, q):
    logp = np.log(p)
    logq = np.array([np.log(e) if e!= 0. else logp[list(q).index(e)] for e in q])
    
    kl = integrate.simpson(p * (logp - logq), x)
    
    return kl
################## METROPOLIS ####################
if metropolis:
    k = 0
    n_accepted = 1
    pbar = tqdm(total=M)
    if write_data: 
        with open(paths_file, 'w') as file:
            with open(actions_file, 'w') as file2:
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
                        if write_data:
                            file.write(' '.join([str(x) for x in path_new]) + '\n')
                            file.write(' '.join([str(x) for x in -path_new]) + '\n')
                            file2.write(str(S_new) + '\n')
                            file2.write(str(S_new) + '\n')
                        pbar.update(1)
                        if n_accepted > n_faulty:
                            wf += histograma(path_new, dx) + histograma(-path_new, dx)
                            wf_norm = integrate.simpson(y=wf, x=x_axis)
                            counts = histograma(path_new, dx)
                            if n_accepted % leap == 0:
                                nice_plot(x_axis,wf/wf_norm, S_paths, n_accepted,
                                       path_new)
                            if n_accepted == M-1:
                                nice_plot(x_axis, wf/wf_norm, S_paths, n_accepted,
                                       path_new, 'mc.pdf', save=False)
                                x = []
                                for i in range(100):
                                    for j in range(counts[i]):
                                        x.append(x_axis[i])
                                        
                                print('\n', scipy.stats.kstest(x, 'norm'))
                  
                    k += 1
            pbar.close()
            file.close()
            file2.close()
    else:
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
                if n_accepted > n_faulty:
                    wf += histograma(path_new, dx) + histograma(-path_new, dx)
                    wf_norm = integrate.simpson(y=wf, x=x_axis)
                    counts = histograma(path_new, dx)
                    if n_accepted % leap == 0:
                        nice_plot(x_axis,wf/wf_norm, S_paths, n_accepted,
                               path_new)
                    if n_accepted == M-1:
                        nice_plot(x_axis, wf/wf_norm, S_paths, n_accepted,
                               path_new, 'mc.pdf', save=False)                                
                        print('\n', kltest(x_axis, y_target, wf/wf_norm))
          
            k += 1
    pbar.close()
"""    
with open('aux_file.txt', 'w') as file:
    for i, j in zip([n for n in range(M)], S_paths):
        file.write(f'{i} {j}\n')
    file.close()
    """
if save_wf:
    with open(f'saved_data/wf_N{N}_M{M}.txt', 'w') as file:
        for x, wf2 in zip(x_axis, wf/wf_norm):
            file.write(f'{x} {wf2}\n')
        file.close()
    print('Wave function data correctly saved.')
if save_plot:
    nice_plot(x_axis, wf/wf_norm, S_paths, n_accepted,
           path_new, 'mc.pdf', save=True)


#%% ############ <X> #############
"""
x,y = [],[]
with open('saved_data/wf_N{N}_M{M}.txt', 'r') as file:
    for line in file.readlines():
        line = line.split(' ')
        x.append(float(line[0]))
        y.append(float(line[1]))
    file.close()
x, wf = np.array(x), np.array(y)
wf_norm = integrate.simpson(y=wf, x=x)

E_X = integrate.simpson(y=wf*x/wf_norm, x=x)
E_X2 = integrate.simpson(y=wf*x**2/wf_norm, x=x)
E_X3 = integrate.simpson(y=wf*x**3/wf_norm, x=x)
E_X4 = integrate.simpson(y=wf*x**4/wf_norm, x=x)
E = m*w**2*E_X2
print('\n')
print(f'<X> = {E_X}')
print(f'<X²> = {E_X2}')
print(f'<X³> = {E_X3}')
print(f'<X⁴> = {E_X4}')
print(f'<E> = {E}')
"""









    


