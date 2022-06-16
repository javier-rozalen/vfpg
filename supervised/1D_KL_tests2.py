######################## IMPORTS ########################
import numpy as np
from tqdm import tqdm
import os,math,torch
import matplotlib.pyplot as plt
from scipy import integrate

# My modules
from modules.neural_networks import *
from modules.plotters import loss_plot, loss_plot_testos
from modules.integration import simpson_weights
from modules.aux_functions import train_loop
from modules.physical_constants import *
from modules.dir_support import dir_support
from modules.loss_functions import *

######################## PARAMETERS ########################
# General parameters
model = 'ff'
M = 5000
N = 1
Nin = N
Nhid = 10
Nout = 1
num_layers = 2
n_epochs = 5000
N_test = M
seed = 1
torch.manual_seed(seed)

mu = torch.tensor(0.)
sigma = torch.tensor(1.)

learning_rate = 1e-2
epsilon = 1e-8

# Plotting parameters
leap = n_epochs/5
adaptive_factor = 1.2
eps = 5  # %

# Booleans
show_periodic_plots = True

######################## TRAIN/TEST SET ########################
# Train set
if N ==1:
    # 1-D
    train_data = torch.linspace(-5,5,M).unsqueeze(1)
    p_train_data = torch.cos(2*train_data).squeeze(1)
    p_train_data = (torch.exp(-0.5*train_data**2)/torch.sqrt(2*torch.tensor(np.pi)))
else:
    # N-D
    train_data = torch.normal(mu, sigma, size=(M, N))
    p_train_data_prime = torch.exp(-0.5*((train_data-mu.expand(M, N))/sigma.expand(
        M, N))**2)/(torch.sqrt(2*torch.tensor(np.pi))*sigma.expand(M, N))
    p_train_data = torch.prod(p_train_data_prime, dim=1).unsqueeze(1)
    
# Test set
x_test = train_data
#x_test = torch.log(torch.linspace(-4,4,M)).unsqueeze(1)
#x_test = torch.normal(mu,sigma,size=(N_test,1))

######################## NEURAL NETWORK STUFF ########################
q_params_nn = q_phi_simple(Nin,Nhid,Nout,num_layers)
loss_fn = loss_DKL_simple
optimizer = torch.optim.RMSprop(params=q_params_nn.parameters(),lr=learning_rate,eps=epsilon)

######################## EPOCHS LOOP ########################
x_axis, loss_list = [], []
for t in tqdm(range(n_epochs)):
    things = train_loop(model=q_params_nn,train_set=train_data,target_data=p_train_data,
                   loss_fn=loss_fn,optimizer=optimizer)
    I = things[0]
    loss_list.append(I.item())
    x_axis.append(t)
    if t == n_epochs-1 or (t+1) % leap == 0:
        #print(I.item())
        if show_periodic_plots and N==1:
            error = 0.01
            color = 'blue' if error <= eps/100 else 'red'
            pred_q = q_params_nn(x_test)
            loss_plot_testos(x_axis,loss_list,I.item(),adaptive_factor,x_test,pred_q.detach())
