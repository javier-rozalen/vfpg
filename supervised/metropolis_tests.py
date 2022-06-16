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
from modules.neural_networks import *

#################################  PARAMETERS #################################
# General parameters
arch = 'ff'
N_ff = 30
mu = 0
sigma = 1/6
M_ff = 10000
m = 1
w = 1
n_faulty = 200
nbins = 100
T = 100
d = 1.
hbar = 1.
h = T/N_ff

# RNN parameters
input_size = 1
hidden_size = 2*input_size
num_layers = 1
Dense = True
M_rnn = 10000
N_rnn = 15

# Plotting parameters
leap = M_ff/20 if arch=='ff' else M_rnn/10

# Trained model parameters
Nhid = 50
num_layers = 2
Nc = 1
learning_rate = 1e-3
seed = 2

# Booleans
metropolis = True
expectations = False

M = M_ff if arch=='ff' else M_rnn
N = N_ff if arch=='ff' else N_rnn

if arch == 'ff':
    trained_model = f'saved_models/ff/N{N}_M{M}_nhid{Nhid}_Nc{Nc}_lr{learning_rate}_nlayers{num_layers}_seed{seed}.pt'
elif arch == 'rnn':
    trained_model = f'saved_models/rnn/N{N}_M{M}_num_layers{num_layers}_Dense{Dense}_lr{learning_rate}_seed{seed}.pt'
paths_file = f'../MonteCarlo/saved_data/paths_N{N}_M{M}.txt'
actions_file = f'../MonteCarlo/saved_data/actions_N{N}_M{M}.txt'

######################## AUXILIARY STUFF ########################
def S_HO(x,h,m,w):
    """
    Euclidean-time action of the 1D, 1-particle H.O.
    
    Parameters
    ----------
    x : list
        (positions of the) Path.

    Returns
    -------
    S : float
        Action of the path given as input.

    """
    S_prime = 0.
    x = x.numpy()[0]
    for i in range(len(x)-1):
        x_i1 = x[i+1]
        x_i = x[i]
        S_prime += ((x_i1-x_i)/h)**2+(w*(x_i1+x_i)/2)**2
        
    return 0.5*m*h*S_prime

######################## DATA FETCHING ########################

print('Fetching data...')
path_manifold,S_manifold = [],[]
with open(paths_file,'r') as file:
    for line in file.readlines():
        path = [torch.tensor(float(x)) for x in line.split(' ')]
        path_manifold.append(torch.stack(path))
    file.close()
with open(actions_file,'r') as file:
    for line in file.readlines():
        S = torch.tensor([float(line)])
        S_manifold.append(S)
    file.close()
x_tensor = torch.stack(path_manifold)
S_tensor = torch.stack(S_manifold)
p_tensor = torch.exp(-S_tensor).unsqueeze(1)
M = len(path_manifold)
N = len(path_manifold[0])

print('Data fetching complete.\n')

###############################################################################
dir_support(['saved_data'])
dx = 10/nbins

# Neural Network loading
with torch.no_grad():
    if arch == 'ff':
        Nin = torch.load(trained_model)['Nin']
        Nhid = torch.load(trained_model)['Nhid']
        Nout = torch.load(trained_model)['Nout']
        num_layers = torch.load(trained_model)['num_layers']
        
        q_phi = q_phi_ff_connected(Nin,Nhid,Nout,num_layers).to('cpu')
        q_phi.load_state_dict(torch.load(trained_model)['model_state_dict'])
        q_phi.eval()
        
    elif arch == 'rnn':
        input_size = torch.load(trained_model)['input_size']
        hidden_size = torch.load(trained_model)['hidden_size']
        num_layers = torch.load(trained_model)['num_layers']
        Dense = torch.load(trained_model)['Dense']
        
        q_phi = q_phi_rnn(input_size,hidden_size,num_layers,Dense).to('cpu')
        q_phi.load_state_dict(torch.load(trained_model)['model_state_dict'])
        q_phi.eval()

x0 = torch.normal(mu,sigma,size=(1,N))
#x0 = [torch.tensor(0.)]*N
#x0 = torch.stack(x0).unsqueeze(0) if arch=='ff' else torch.stack(x0).view(1,N,-1)
x0 = torch.zeros(size=(1,N))
paths = [x0]
#q_paths = [q_phi(p_tensor[0],x0)[0]]
q_paths = [S_HO(x0,h,m,w)]
q_paths = [q_phi(p_tensor[0],x0)[4]]
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
                chi = torch.normal(mu,sigma,size=(1,N))
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










    


