######### IMPORTS ##########
import torch,math
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from modules.physical_constants import m, w, hbar
from modules.lagrangians import L_HO
###########################

############################ AUXILIARY FUNCTIONS ##############################
def loss_DKL(model, train_set=0, target_set=0, h=1):
    """
    Loss function.

    Returns
    -------
    loss : tensor
        Kullback-Leibler divergence.

    """
    
    # We sample the paths and their probs
    paths, paths_cond_probs = model(train_set) # paths.size() = [M, N]
    
    # We compute the actions of all the paths
    M = torch.tensor(paths.size()[0])
    S_paths = []
    for path in paths:
        #print(path)
        L_path = L_HO(path, h, m, w) # lagrangian of the path
        #print(h.repeat(path.size(0)))
        S_path = torch.dot(L_path, h.repeat(path.size(0))) # Action of the path
        #print(S_path)
        S_paths.append(S_path)
    S_paths = torch.stack(S_paths).squeeze()
    #print(f'S_paths: {S_paths}')
    
    # Monte Carlo estimate of KL divergence
    #print(f'Sum of logs: {torch.sum(torch.log(paths_cond_probs), dim=1)}')
    f = (1/hbar)*S_paths + torch.sum(torch.log(paths_cond_probs), dim=1)
    loss_KL = (1/M)*torch.sum(f)
    MC_err = torch.sqrt((1/M)*torch.sum(f**2)-loss_KL**2) / torch.sqrt(M)
    #print(f'loss_KL: {loss_KL}')

    return loss_KL, MC_err, paths
    
    
    
    
    