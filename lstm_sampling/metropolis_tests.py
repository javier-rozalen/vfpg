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
from modules.neural_networks import VFPG_ours, VFPG_theirs
from modules.plotters import metropolis_histo

#################################  PARAMETERS #################################
# General parameters
which_net = 'theirs'
Nc = 5

# Metropolis parameters
mu = 0.
sigma = 1/6
n_faulty = 200
d = 1.

# LSTM parameters
dev = 'cpu'
M = 1
N = 5
input_size = 2 if which_net == 'theirs' else 1 # dimension of the input 
nhid = 6 # number of hidden neurons
hidden_size = 7 # dimension of the LSTM hidden state vector
out_size = (1 + 2)*Nc # size of the LSTM output, y
num_layers = 1 # number of stacked LSTM layers
Dense = True # controls wether there is a Linear layer after the LSTM or not

# Hyperparameters
learning_rate = 1e-5
seed = 1

# Plotting parameters
nbins = 100
leap = M / 20 

# Target pdf params
mus, sigmas = torch.zeros(N), torch.ones(N)

# Booleans
metropolis = True
expectations = False

torch.manual_seed(seed)

######################## AUXILIARY STUFF ########################
dx = 10/nbins
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

######################## DATA FETCHING ########################
print('Fetching data...')
# Neural Network loading
with torch.no_grad():
    q_phi = VFPG_theirs(dev=dev,
                        M=M,
                        N=N,
                        input_size=input_size,
                        nhid=nhid,
                        hidden_size=hidden_size,
                        out_size=out_size,
                        num_layers=num_layers,
                        Dense=Dense)
    
    q_phi.load_state_dict(torch.load('saved_models/model.pt')['state_dict'])
    q_phi.eval()

print('Data fetching complete.\n')

###################### METROPOLIS INITIALISATION ######################

x0 = torch.zeros(size=(1, N)).repeat(M, 1, 1)
print(f'\nx0: {x0}', x0.shape)
q_x0 = q_phi(x0)

paths = [x0.squeeze(1)]
q_paths = [q_x0]

wf = np.array([0.]*nbins)

################## METROPOLIS ####################
if metropolis:
    k = 0
    n_accepted = 1
    pbar = tqdm(total=M)
    while n_accepted<M:
        chi = torch.normal(mu, sigma, size=(M, N, 1))
        #chi = np.random.normal(mu,sigma,N)
        #chi = chi if arch=='ff' else chi.view(1,N,-1)
        path_old = paths[-1]
        path_new = path_old+d*chi
        path_new[-1] = path_new[0]
        #path_new = path_new if arch=='ff' else path_new.view(1,N,-1)
        
        # Neural net
        p_new = q_phi(p_tensor[k+1],path_new)[4]
        p_old = q_phi(p_tensor[k],path_old)[4]
        
        # Exact function
        #p_new = np.exp(-S_HO(path_new,h,m,w))
        #p_old = np.exp(-S_HO(path_old,h,m,w))
        
        path_new[-1]=path_new[0]
        A = min([1,p_new/p_old])
        u = torch.rand(1)
        
        if u<=A:
            accepted = True
        else:
            accepted = False
        """
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
        """        
        if accepted:
            n_accepted += 1
            paths.append(path_new)
            q_paths.append(-torch.log(p_new))
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
    











    


