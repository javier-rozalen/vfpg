######### IMPORTS ##########
import torch,math
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
###########################

############################ AUXILIARY FUNCTIONS ##############################
def loss_DKL(model,train_set=0,target_set=0):
    """
    Loss function.

    Returns
    -------
    loss : tensor
        Kullback-Leibler divergence.

    """
    
    # We sample the paths, their probs and their actions 
    x,q_params = model(train_set)
    
    # Monte Carlo
    M = x.size()[0]
    I = (1/M)*torch.sum(logp-logp_nn)
    
    # Endpoints constraints
    sumand_i = (q_params['mu'][0]-torch.tensor(x_i))**2+q_params['sigma'][0]**2
    sumand_f = (q_params['mu'][-1]-torch.tensor(x_f))**2+q_params['sigma'][-1]**2
    
    loss_kl = I
    loss_i = N*torch.sum(sumand_i)
    loss_f = N*torch.sum(sumand_f)
    
    loss = loss_kl+loss_i+loss_f
    return loss
    
    
    
    
    