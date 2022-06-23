######### IMPORTS ##########
import torch,math
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from modules.physical_constants import m, w, hbar
from modules.lagrangians import L_HO
###########################

############################ AUXILIARY FUNCTIONS ##############################
def loss_DKL(model, train_set=[], target_set=[], h=1):
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
    L_paths = L_HO(paths, h, m, w) # lagrangiangs of all paths
    S_paths = h * torch.sum(L_paths, dim=1) # actions of all paths
    
    # Monte Carlo estimate of KL divergence
    M = torch.tensor(paths.size(0))
    q_phi = torch.sum(torch.log(paths_cond_probs), dim=1)
    f = (1/hbar)*S_paths + q_phi
    loss_KL = (1/M)*torch.sum(f)
    MC_err = torch.sqrt((1/M)*torch.sum(f**2)-loss_KL**2) / torch.sqrt(M)

    return loss_KL, MC_err, paths

def loss_MSE(model, train_set=[], target_set=[], h=1):
    """
    Mean Squared Error Loss Function.

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    train_set : TYPE, optional
        DESCRIPTION. The default is [].
    target_set : TYPE, optional
        DESCRIPTION. The default is [].
    h : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    MSE Loss of the given model at the given train_set. 

    """
    
    # We sample the paths and their probs
    paths, paths_cond_probs = model(train_set) # paths.size() = [M, N]
    
    # We compute the actions of all the paths
    M = torch.tensor(paths.size(0))
    S_paths = []
    for path in paths:
        #print(path)
        L_path = L_HO(path, h, m, w) # lagrangian of the path
        print(L_path, L_path.shape)
        S_path = h * torch.sum(L_path) # Action of the path
        #print(S_path)
        S_paths.append(S_path)
    S_paths = torch.stack(S_paths).squeeze()
    #print(f'S_paths: {S_paths}')
    
    #MSE
    p_E = torch.tensor(torch.exp(-S_paths/hbar), dtype=torch.float32)
    q_phi = torch.sum(torch.log(paths_cond_probs), dim=1)
    q_phi = torch.prod(paths_cond_probs, dim=1)
    loss_MSE = nn.MSELoss()
    loss = loss_MSE(p_E, q_phi)
    
    return loss, torch.tensor(0.), paths
    
    