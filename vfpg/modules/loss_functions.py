######### IMPORTS ##########
import torch,math
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from modules.physical_constants import m, w, hbar
from modules.lagrangians import L_HO, L_double_well
###########################

############################ AUXILIARY FUNCTIONS ##############################
def loss_DKL(model, train_set=[], target_set=[], potential=L_HO, h=1, x_i=0., 
             x_f=1.):
    """
    Kullback-Leibler divergence.

    Parameters
    ----------
    model : (instance of) class
        Neural Network.
    train_set : tensor, optional
        Training set. The default is [].
    target_set : tensor, optional
        Labels of training set. The default is [].
    potential: function, optional
        The potential. The default is L_HO.
    h : tensor or float, optional
        time grid cell size. The default is 1.

    Returns
    -------
    loss_KL : tensor
        Kullback-Leibler divergence.
    MC_err : tensor
        MonteCarlo integration error estimate.
    paths : tensor
        Path manifold.

    """
    
    # We sample the paths and their probs
    paths, paths_cond_probs, mus, sigmas = model(train_set) # paths.size() = [M, N]
    
    # We compute the actions of all the paths
    L_paths = L_HO(paths, h, m, w) # lagrangiangs of all paths, size [M, N]
    S_paths = h * torch.sum(L_paths, dim=1) # actions of all paths, size [M, N]
    S_paths = S_paths.squeeze(1)
    
    # Loss_KL
    M = torch.tensor(paths.size(0))
    log_q_phi = torch.sum(torch.log(paths_cond_probs), dim=1)
    f = (1/hbar)*S_paths + log_q_phi
    loss_KL = (1/M)*torch.sum(f)
    MC_err = torch.sqrt((1/M)*torch.sum(f**2)-loss_KL**2) / torch.sqrt(M)
    
    # Loss_initial
    N = paths.size(1)
    mus_i = mus[:, 0, :]
    sigmas_i = sigmas[:, 0, :]
    iss = N*torch.sum((mus_i-x_i.expand(mus_i.shape))**2 + sigmas_i**2, dim=1)
    loss_i = torch.sum(iss)
    
    # Loss_final
    mus_f = mus[:, -1, :]
    sigmas_f = sigmas[:, -1, :]
    fss = N*torch.sum((mus_f-x_f.expand(mus_f.shape))**2 + sigmas_f**2, dim=1)
    loss_f = torch.sum(fss) 
    """
    print(f'loss_KL: {loss_KL}')
    print(f'loss_i: {loss_i}')
    print(f'loss_f: {loss_f}')
    """
    # Total loss
    loss = loss_KL + loss_i + loss_f
    """
    print(f'paths_cond_probs: {paths_cond_probs}', paths_cond_probs.shape)
    print(f'q_phi: {q_phi}', q_phi.shape)
    print(f'S_paths: {S_paths}', S_paths.shape)
    print(f'f: {f}', f.shape)
    """
    return loss_KL, MC_err, paths

def loss_MSE(model, train_set=[], target_set=[], h=1):
    """
    Mean Squared Error Loss Function.

    Parameters
    ----------
    model : (instance of) class
        Neural Network.
    train_set : tensor, optional
        Training set. The default is [].
    target_set : tensor, optional
        Labels of training set. The default is [].
    h : tensor or float, optional
        time grid cell size. The default is 1.

    Returns
    -------
    loss : tensor
        MSE loss.
    

    """
    
    # We sample the paths and their probs
    paths, paths_cond_probs = model(train_set) # paths.size() = [M, N]
    
    # We compute the actions of all the paths
    L_paths = L_HO(paths, h, m, w) # lagrangiangs of all paths, size [M, N]
    S_paths = h * torch.sum(L_paths, dim=1) # actions of all paths, size [M, N]
    S_paths = S_paths.squeeze(1)
    
    # MSE
    p_E = torch.tensor(torch.exp(-S_paths/hbar), dtype=torch.float32)
    q_phi = torch.sum(torch.log(paths_cond_probs), dim=1)
    q_phi = torch.prod(paths_cond_probs, dim=1)
    loss_MSE = nn.MSELoss()
    loss = 1e11*loss_MSE(p_E, q_phi)
    
    return loss, torch.tensor(0.), paths
    
    