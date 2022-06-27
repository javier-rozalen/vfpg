######### IMPORTS ##########
import torch,math
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
###########################

############################ AUXILIARY FUNCTIONS ##############################
def loss_DKL(model, train_set, mus, sigmas): 
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
    paths, paths_cond_probs = model(train_set) # paths.size() = [M, N]
    """
    print(f'paths: {paths}', paths.shape)
    print(f'paths_cond_probs: {paths_cond_probs}', paths_cond_probs.shape)
    """
    # Loss_KL
    M, N = torch.tensor(paths.size(0)), torch.tensor(paths.size(1))
    
    mus_ext, sigmas_ext = mus.repeat(M, 1), sigmas.repeat(M, 1)
    f = torch.sum(torch.log(paths_cond_probs)+0.5*((paths.squeeze()-mus_ext)/sigmas_ext)**2, dim=1)
    loss_KL_var = (1/M)*torch.sum(f) 
    loss_KL = loss_KL_var + 0.5*N*torch.log(2*torch.tensor(np.pi)) + torch.sum(sigmas)
    MC_err = torch.sqrt((1/M)*torch.sum(f**2)-loss_KL_var**2) / torch.sqrt(M)
    """
    print(f'paths: {paths}', paths.shape)
    print(f'paths-mus_ext: {(paths-mus_ext)/sigmas_ext}', (paths-mus_ext).shape)
    print(f'f: {f}', f.shape)
    """
    return loss_KL, MC_err, paths

def loss_MSE(model, train_set, mus, sigmas):
    
    # We sample the paths and their probs
    paths, paths_cond_probs = model(train_set) # paths.size() = [M, N]
    
    # Loss MSE
    mse = nn.MSELoss()
    q_phi = torch.prod(paths_cond_probs, dim=1)
    M, N = torch.tensor(paths.size(0)), torch.tensor(paths.size(1))
    mus_ext, sigmas_ext = mus.repeat(M, 1), sigmas.repeat(M, 1)
    p = torch.exp(-0.5*torch.sum(((paths.squeeze()-mus_ext)/sigmas_ext)**2, dim=1))
    p /= (torch.tensor(2*np.pi))**(N/2)
    p /= torch.prod(sigmas)
    loss = mse(p, q_phi)
    
    return loss, torch.tensor(0.), paths
    

    
    