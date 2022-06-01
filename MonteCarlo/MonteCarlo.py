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

# My Modules
from modules.actions import S_HO
from modules.plotters import histo2
from modules.dir_support import dir_support

################################# GENERAL PARAMETERS ##########################
seed = 1
N = 50 
mu = 0
sigma = 1/6
M = 100000
leap = M/20
m = 1
w = 1
n_faulty = 100
T = 100
d = 1.
dx = 0.1
hbar = 1.
metropolis = True
save_data = True

###############################################################################
dir_support(['saved_data'])
np.random.seed(seed)
h = T/N

x0 = [0.]*N
x0 = np.random.normal(0,1,N).tolist()
paths = [x0]
S_paths = [S_HO(x0,h,m,w)]
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
                count[j] += 1
                done = True
            else:
                j += 1
    return np.array(count)

################## METROPOLIS ####################
if metropolis:
    k = 0
    n_accepted = 1
    pbar = tqdm(total=M)
    with open(f'saved_data/paths_N{N}_M{M}.txt','w') as file:
        with open(f'saved_data/actions_N{N}_M{M}.txt','w') as file2:
            while n_accepted<M:
                chi = np.random.normal(mu,sigma,N)
                path_old = paths[-1]
                path_new = path_old+d*chi
                path_new[-1]=path_new[0]
                S_new = S_HO(path_new,h,m,w)
                S_old = S_HO(path_old,h,m,w)
                delta_S = S_new-S_old
                
                if delta_S<=0:
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
                    file.write(' '.join([str(x) for x in path_new])+'\n')
                    file.write(' '.join([str(x) for x in -path_new])+'\n')
                    file2.write(str(S_new)+'\n')
                    file2.write(str(S_new)+'\n')
                    pbar.update(1)
                    if n_accepted > n_faulty:
                        wf = wf + histograma(path_new,dx)+histograma(-path_new,dx)
                        if n_accepted%leap == 0:
                            x_axis = np.linspace(-4.95,4.95,100)
                            wf_norm = integrate.simpson(y=wf,x=np.linspace(-4.95,4.95,100))
                            histo2(x_axis,wf/wf_norm,S_paths,n_accepted,path_new)
                        if n_accepted == M-1:
                            histo2(x_axis,wf/wf_norm,S_paths,n_accepted,path_new,'mc.pdf',save=True)
              
                k += 1
        pbar.close()
        file.close()
        file2.close()
    
if save_data:
    # We save the wave function data
    with open('saved_data/wf_N{N}_M{M}.txt','w') as file:
        for x,y in zip(x_axis,wf/wf_norm):
            file.write(str(x)+' '+str(y)+'\n')
        file.close()
    print('\nData saved at saved_data/wf_N{N}_M{M}.txt')
    
#%% ############ <X> #############
x,y = [],[]
with open('saved_data/wf_N{N}_M{M}.txt','r') as file:
    for line in file.readlines():
        line = line.split(' ')
        x.append(float(line[0]))
        y.append(float(line[1]))
    file.close()
x,wf=np.array(x),np.array(y)
wf_norm = integrate.simpson(y=wf,x=x)

E_X = integrate.simpson(y=wf*x/wf_norm,x=x)
E_X2 = integrate.simpson(y=wf*x**2/wf_norm,x=x)
E_X3 = integrate.simpson(y=wf*x**3/wf_norm,x=x)
E_X4 = integrate.simpson(y=wf*x**4/wf_norm,x=x)
E = m*w**2*E_X2
print('\n')
print(f'<X> = {E_X}')
print(f'<X²> = {E_X2}')
print(f'<X³> = {E_X3}')
print(f'<X⁴> = {E_X4}')
print(f'<E> = {E}')










    


