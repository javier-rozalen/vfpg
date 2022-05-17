#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:08:20 2022

@author: jozalen
"""

import torch,math
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import grad

hbar = 197.32968 # MeV·fm

def simpson_weights(x):
    x = list(x)
    N = len(x)
    w_k = []
    for k in range(N):
        if k == 0:
            w_k.append(0.5*(x[1]-x[0]))
        elif k == N-1:
            w_k.append(0.5*(x[-1]-x[-2]))
        else:
            w_k.append(0.5*(x[k+1]-x[k-1]))   
    
    return torch.tensor(w_k)

#%% GAUSSIAN VARIABLE GENERATION
mean = torch.tensor(0.0,dtype=torch.double)
std = torch.tensor(1.0,dtype=torch.double)

z = torch.normal(mean=mean,std=std)

#%% S_E[x(t)] COMPUTATION
torch.manual_seed(1)
plot_path = True
m = 1.0
f = 0.5
w = 2*math.pi*f

# sample path
N = 50
x_0 = 10.
t_0 = 0.
t_f = 5.

############################### THE ANN ######################################
# We create our nn class as a child class of nn.Module
device = 'cpu'

Nin = 1
Nhid = 1000
Nout = Nin
lr = 1e-1

# ANN Parameters
W1 = torch.rand(Nhid,Nin,requires_grad=True,dtype=torch.double)*(1.) # First set of coefficients
B = torch.rand(Nhid,dtype=torch.double)*2.-torch.tensor(1.) # Set of bias parameters
W2 = torch.rand(Nout,Nhid,requires_grad=True,dtype=torch.double) # Second set of coefficients
class NeuralNetwork(nn.Module):
    def __init__(self):       
        super(NeuralNetwork, self).__init__()
         
        # We set the operators 
        self.lc1 = nn.Linear(Nin,Nhid,bias=True) # shape = (Nhid,Nin)
        self.actfun = nn.Softplus() # activation function
        self.lc2 = nn.Linear(Nhid,Nout,bias=False) # shape = (Nout,Nhid)
        
        # We set the parameters 
        with torch.no_grad():
            self.lc1.weight = nn.Parameter(W1)
            self.lc1.bias = nn.Parameter(B)
            self.lc2.weight = nn.Parameter(W2)
   
    # We set the architecture
    def forward(self, x): 
        o = self.actfun(self.lc1(x))
        o = self.lc2(o)
        return o
    
# We load our psi_ann to the CPU (or GPU)
psi_ann = NeuralNetwork().to(device) 
t_steps_arange = torch.arange(start=t_0,end=t_f,step=(t_f-t_0)/N,dtype=torch.double,requires_grad=True)
t_steps_linspace = torch.linspace(start=t_0,end=t_f,steps=N,dtype=torch.double,requires_grad=True)
t_steps = [torch.tensor(e,dtype=torch.double,requires_grad=True).unsqueeze(0) for e in np.linspace(t_0,t_f,N)]

weights = simpson_weights(t_steps)

def cost():
    global t_steps,x_steps,S,L,v_steps
    
    x_steps,v_steps=[],[]
    for i in range(N):
        x_steps.append(psi_ann(t_steps[i]))
        v_steps.append(torch.tensor(grad(outputs=x_steps[i],inputs=t_steps[i],create_graph=True)))
    
    x_steps = torch.stack(x_steps).squeeze(1)
    v_steps = torch.stack(v_steps).squeeze(1) 
    
    # lagrangian
    L=0.5*m*(v_steps**2-(w*x_steps)**2)

    # action
    S = torch.dot(L,weights)
    return -S+(x_steps[0]-5.)**2

# train loop
loss = cost
optimizer = torch.optim.RMSprop(params=psi_ann.parameters(),lr=lr,eps=1e-8)
torch.autograd.set_detect_anomaly(True)

epochs = 100000
leap = 500
for i in range(epochs):
    optimizer.zero_grad()
    loss().backward()
    optimizer.step()
    if (i+1)%leap == 0:
        print('Epoch {}'.format(i+1))
        print('S = {:.3f}'.format(S))
        if plot_path:
                fig,ax = plt.subplots(nrows=1,ncols=1)
                
                ax.set_title(r'Path $x(\tau)$')
                ax.set_xlabel(r'$\tau$')
                ax.set_ylabel('$x$',rotation=180)
                ax.plot([e.item() for e in t_steps],x_steps.detach().numpy())
                
                plt.show()
                

