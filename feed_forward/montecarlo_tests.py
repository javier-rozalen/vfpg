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
import torch

# My Modules
from modules.plotters import histo2
from modules.dir_support import dir_support
from modules.neural_networks import show_layers
from modules.neural_networks import q_phi_layers as neural_net_layers

#################################  PARAMETERS #################################
# General parameters
N = 50 
mu = 0
sigma = 1/6
M = 10000
leap = M/10
m = 1
w = 1
n_faulty = 0
nbins = 100
T = 100
d = 1.
hbar = 1.

# Trained model parameters
Nhid = 20
num_layers = 2
learning_rate = 1e-2
seed = 5

metropolis = True
expectations = False

trained_model = f'saved_models/nhid{Nhid}_lr{learning_rate}_nlayers{num_layers}_seed{seed}.pt'
paths_file = '../MonteCarlo/saved_data/paths.txt'

######################## DATA FETCHING ########################
print('Fetching data...')
path_manifold = []
with open(paths_file,'r') as file:
    for line in file.readlines():
        path = [torch.tensor(float(x)) for x in line.split(' ')]
        path_manifold.append(torch.stack(path))
    file.close()
x_tensor = torch.stack(path_manifold)
print('Data fetching complete.\n')

###############################################################################
dir_support(['saved_data'])
dx = 10/nbins

# Neural Network loading
with torch.no_grad():
    Nin,Nhid,Nout = torch.load(trained_model)['Nin'],torch.load(trained_model)['Nhid'],torch.load(trained_model)['Nout']
    q_phi = neural_net_layers(Nin,Nhid,Nout,num_layers).to('cpu')
    q_phi.load_state_dict(torch.load(trained_model)['model_state_dict'])
    #show_layers(q_phi)
    q_phi.eval()

x0 = [torch.tensor(0.)]*N
x0 = torch.stack(x0).unsqueeze(0)
x0 = torch.normal(mu,sigma,size=(1,N))

paths = [x0]
q_paths = [q_phi(x0)]
wf = np.array([0.]*nbins)

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
    count = [0]*nbins
    n = len(x)
	
    for i in range(n):

        j = 0
        done = False
        while -5 + j*dx <= +5-dx and done == False:
			
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
                chi = torch.normal(mu,sigma,size=(1,N)).squeeze(0)
                path_old = paths[-1]
                path_new = path_old+d*chi
                path_new[-1]=path_new[0]
                S_new = q_phi(path_new)
                S_old = q_phi(path_old)
                A = min([1,S_new.item()/S_old.item()])
                u = torch.rand(1)
                
                if u<=A:
                    accepted = True
                else:
                    accepted = False
                  
                if accepted:
                    n_accepted += 1
                    paths.append(path_new)
                    """
                    q_paths.append(S_new)
                    file.write(' '.join([str(x) for x in path_new])+'\n')
                    file.write(' '.join([str(x) for x in -path_new])+'\n')
                    file2.write(str(S_new)+'\n')
                    file2.write(str(S_new)+'\n')
                    """
                    pbar.update(1)
                    if n_accepted > n_faulty:
                        wf = wf + histograma(path_new.numpy()[0],dx)+histograma(-path_new.numpy()[0],dx)
                        if n_accepted%leap == 0:
                            x_axis = np.linspace(-4.95,4.95,100)
                            wf_norm = integrate.simpson(y=wf,x=np.linspace(-4.95,4.95,100))
                            histo2(x_axis,wf/wf_norm,q_paths,n_accepted,path_new.detach().numpy()[0],
                                   Nhid,num_layers,learning_rate)
                            #print(n_accepted/k)
              
                k += 1
        pbar.close()
        file.close()
        file2.close()
    
# We save the wave function data
with open('saved_data/wf_N{N}_M{M}.txt','w') as file:
    for x,y in zip(x_axis,wf/wf_norm):
        file.write(str(x)+' '+str(y)+'\n')
    file.close()
    
#%% ############ <X> #############
if expectations:
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










    


