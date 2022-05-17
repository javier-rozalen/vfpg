#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:01:52 2022

@author: jozalen
"""

import torch,math,time
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import grad
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from modules.vfpg import VFPG
from modules.integration import simpson_weights
from modules.lagrangians import L_HO
import multiprocessing as mp

def show_layers(model):
    print("\nLayers and parameters:\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:100]} \n")

def q(x,q_params):
    """
    Computes the probability of input path x using the input parameters q_params.

    Parameters
    ----------
    x : list
        path generated by VFPG.
    q_params : dict
        GMM parameters at each time step (3*Nc at each t).

    Returns
    -------
    prod : tensor
        probability of input path x.

    """
    prod = torch.tensor(1.)
    c = 0
    for x_k in x:
        gamma,mu,sigma = q_params['gamma'][c],q_params['mu'][c],q_params['sigma'][c]
        normals = torch.tensor([torch.exp(-0.5*((x_k-mu_i)/sigma_i)**2)/(sigma_i*torch.sqrt(torch.tensor(2*math.pi))) for mu_i,sigma_i in zip(mu,sigma)])
        q_k = torch.dot(gamma,normals)
        assert q_k>=0
        prod *= q_k
        c += 1

    return prod

def loss():
    """
    Loss function.

    Returns
    -------
    loss : tensor
        Kullback-Leibler divergence.

    """
    
    global path_manifold,S_manifold,loss,loss_kl,loss_i,loss_f,lnz,error,I,S,f
    global S_mean,q_mean
    path_manifold = []
    q_manifold = []
    S_manifold = []
    
    # We sample the paths, their probs and their actions 
    for _ in range(n_paths):
        z = torch.randn(input_size,dtype=torch.float32)
        x,q_params = vfpg(z)
        f = q(x,q_params)
        S = torch.dot(L_HO(x,t,m,w),int_weights)
        # Appends
        path_manifold.append(x)
        q_manifold.append(f)
        S_manifold.append(S)
        
    q_tensor = torch.stack(q_manifold)
    S_tensor = torch.stack(S_manifold)

    # Monte Carlo integration 
    I = (1/n_paths)*torch.sum(torch.log(q_tensor)+(1/hbar)*S_tensor)
    I2 = (1/n_paths)*torch.sum((torch.log(q_tensor)+(1/hbar)*S_tensor)**2)
    error = (1/math.sqrt(n_paths))*torch.sqrt(I2-I**2)
    
    """
    # log(Z) term 
    exps = torch.exp(-S_tensor/hbar)
    Z_tensor = torch.sum(exps)
    lnz = torch.log(Z_tensor)
    """
    
    # Endpoints constraints
    sumand_i = (q_params['mu'][0]-torch.tensor(x_i))**2+q_params['sigma'][0]**2
    sumand_f = (q_params['mu'][-1]-torch.tensor(x_f))**2+q_params['sigma'][-1]**2
    
    loss_kl = I
    loss_i = N*torch.sum(sumand_i)
    loss_f = N*torch.sum(sumand_f)
    
    loss = loss_kl+loss_i+loss_f
    return loss


        
#%% Training example

# General parameters
n_cores = mp.cpu_count()
torch.manual_seed(1)
N = 20 # length of the path
x_i = 0.
x_f = 0.
t_0 = 0.
t_f = 1.
t = [torch.tensor(e) for e in np.linspace(t_0,t_f,N)]
h = t[1]-t[0]
n_paths = 200
int_weights = simpson_weights(t)
#logz = False

# Physical constants
hbar = 1.0
m = 1.0 # mass
f = t_f # frequency
w = 2*math.pi*f

# VFPG parameters
input_size = 2 # dimension of the input random vectors z
Dense = True # controls wether there is a Linear layer after the LSTM or not
Nc = 15 # number of Gaussian components
num_layers = 1 # number of stacked LSTM layers
hidden_size = 20 # dimension of the LSTM output vector
if Dense == False:
    hidden_size = 3*Nc 
    
vfpg = VFPG(N=N,input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,Nc=Nc,Dense=Dense)

# Training parameters
learning_rate = 1e-2
epsilon = 1e-8
smoothing_constant = 0.9
loss_fn = loss
optimizer = torch.optim.SGD(params=vfpg.parameters(),lr=learning_rate)
#optimizer = torch.optim.RMSprop(params=vfpg.parameters(),lr=learning_rate,eps=epsilon)

def train_loop(loss_fn,optimizer):
    """
    Training loop.

    Parameters
    ----------
    loss_fn : function
        loss function.
    optimizer : torch.optim
        optimizer.

    Returns
    -------
    None.

    """  
    optimizer.zero_grad()
    loss_fn().backward()
    optimizer.step()

def pic_creator(bound,j):
    """
    Plots the paths in path_manifold present at current execution time as well as 
    the loss funciton.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(14,6))
    
    ax0 = ax[0]
    ax0.set_title(r'Path $x(\tau)$',fontsize=17)
    ax0.set_xlabel(r'$\tau$',fontsize=17)
    ax0.set_ylabel('$x$',rotation=180,labelpad=10,fontsize=17)
    ax0.tick_params(axis='both',labelsize=15)
    c = 0
    for path,action in zip(path_manifold,S_manifold):
        if c<bound:
            ax0.plot([e.item() for e in t],[e.item() for e in path],label=f'S = {round(action.item(),3)}')
        c+=1    
        
    #ax0.legend(loc='center',bbox_to_anchor=(0.5,-0.35))
    
    ax1 = ax[1]
    ax1.set_title('Loss',fontsize=17)
    ax1.set_xlabel('Epoch',fontsize=17)
    ax1.tick_params(axis='both',labelsize=15)
    ax1.plot([e for e in range(j+1)],loss_list,label='$\mathcal{L}$')
    ax1.plot([e for e in range(j+1)],loss_kl_list,label='$\mathcal{L}_{KL}-lnZ$')
    ax1.plot([e for e in range(j+1)],loss_i_list,label='$\mathcal{L}_i$')
    ax1.plot([e for e in range(j+1)],loss_f_list,label='$\mathcal{L}_f$')    
    ax1.axhline(0.,color='red',linestyle='dashed')
    ax1.legend(fontsize=16)
    
    plt.show()

save_model = False
epochs = 50
leap = 2
bound = 10
loss_list = []
loss_kl_list = []
loss_i_list = []
loss_f_list = []
initial_time = time.time()
for j in range(epochs):
    # Train loop
    train_loop(loss_fn,optimizer)
    
    # Loss tracking
    loss_list.append(loss.item())
    loss_kl_list.append(loss_kl.item())
    loss_i_list.append(loss_i.item())
    loss_f_list.append(loss_f.item())
    
    # Periodic plots + console info
    if (j+1)%leap == 0:
        #assert abs((error/I).item()) <= eps
        print(f'\nEpoch {j+1}\n---------------------------------------')
        print(f'MC relative error: {abs((error/I).item())}')
        print(f'Simpson relative error (S): {(h**4)/abs(S)}')
        print(f'Current time: {time.time()-initial_time}')
        pic_creator(bound,j)
print(f'Done! Time: {time.time()-initial_time}')

# Save the model
if save_model:
    state = {
        'epoch': epochs,
        'state_dict': vfpg.state_dict(),
        'optimizer': optimizer.state_dict()
        }
    model_name = 'first_models.pt'
    torch.save(state,model_name)
    print(f'Model correctly saved at: {model_name}')
    
"""
model.load_state_dict(state['state_dict'])
optimizer.load_state_dict(state['optimizer'])
"""









