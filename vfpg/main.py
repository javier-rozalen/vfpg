#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################# IMPORTS #############################
import torch,math,time
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.autograd import grad
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

# My modules
from modules.physical_constants import *
from modules.neural_networks import *
from modules.loss_functions import *
from modules.aux_functions import *
from modules.integration import simpson_weights
from modules.lagrangians import L_HO
from modules.plotters import *

############################# GENERAL PARAMETERS #############################
# General parameters
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
z_initial = torch.tensor(2.)

# Physical quantities
f = t_f # frequency
w = 2*math.pi*f

# Neural network parameters
input_size = 2 # dimension of the input random vectors z
Dense = True # controls wether there is a Linear layer after the LSTM or not
Nc = 15 # number of Gaussian components
num_layers = 1 # number of stacked LSTM layers
hidden_size = 20 # dimension of the LSTM output vector
if Dense == False:
    hidden_size = 3*Nc 

# Hyperparameters
learning_rate = 1e-2
epsilon = 1e-8
smoothing_constant = 0.9

# Training parameters
epochs = 50
mini_batching = False

# Plotting parameters
leap = 2
bound = 10

# Booleans
save_model = False

############################# AUXILIARY FUNCTIONS #############################
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

    
vfpg = VFPG(N=N,input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,Nc=Nc,Dense=Dense)    
loss_fn = loss_DKL
optimizer = torch.optim.RMSprop(params=vfpg.parameters(),lr=learning_rate,eps=epsilon)

############################# EPOCH LOOP #############################
loss_list = []
loss_kl_list = []
loss_i_list = []
loss_f_list = []
for j in tqdm(range(epochs)):
    # Train loop
    z = torch.randn(input_size,dtype=torch.float32)
    I = train_loop(vfpg,loss_fn,optimizer,train_set=z)
    
    # Loss tracking
    loss_list.append(loss.item())
    loss_kl_list.append(loss_kl.item())
    loss_i_list.append(loss_i.item())
    loss_f_list.append(loss_f.item())
    
    # Periodic plots + console info
    if (j+1)%leap == 0:
        #assert abs((error/I).item()) <= eps
        #print(f'MC relative error: {abs((error/I).item())}')
        #print(f'Simpson relative error (S): {(h**4)/abs(S)}')
        pic_creator(bound,j)
print(f'Done! :)')

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









