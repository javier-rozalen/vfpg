######### IMPORTS ##########
import torch, time
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from modules.physical_constants import m, w, hbar
from modules.lagrangians import L_HO, L_double_well
###########################

toggle_prints = False

def print_(*message):
    if toggle_prints:
        print(*message)
        
print_('\n')
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
    L_paths = potential(paths, h, m, w) # lagrangiangs of all paths, size [M, N]
    S_paths = h * torch.sum(L_paths, dim=1) # actions of all paths, size [M, N]
    S_paths = S_paths.squeeze()
    """
    print(f'L_paths: {L_paths}', L_paths.shape)
    print(f'S_paths: {S_paths}', S_paths.shape)
    """
    # Loss_KL
    M = torch.tensor(paths.size(0))
    log_q_phi = torch.sum(torch.log(paths_cond_probs), dim=1)
    f = (1/hbar)*S_paths + log_q_phi
    loss_KL = (1/M)*torch.sum(f)
    MC_err = torch.sqrt((1/M)*torch.sum(f**2)-loss_KL**2) / torch.sqrt(M)
    """
    print(f'paths_cond_probs: {paths_cond_probs}', paths_cond_probs.shape)
    print(f'log_q_phi: {log_q_phi}', log_q_phi.shape)
    print(f'f: {f}', f.shape)
    print(f'loss_KL: {loss_KL}', loss_KL.shape)
    """
    
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
    return loss, MC_err, paths

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
    
def loss_DKL_v2(model, train_set=[], target_set=[], potential=L_HO, M_MC=100, 
                h=1, x_i=0., x_f=1.):
    """
    Kullback-Leibler divergence v2.

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
    
    # We get the parameters of our variational pdf
    gammas, mus, sigmas = model(train_set) # each of size [batch_size, N]
    
    N = gammas.size(1)
    # Loss_initial
    mus_i = mus[:, 0]
    sigmas_i = sigmas[:, 0]
    loss_i = N*torch.sum((mus_i-x_i.expand(mus_i.shape))**2 + sigmas_i**2)
    
    # Loss_final
    mus_f = mus[:, -1]
    sigmas_f = sigmas[:, -1]
    loss_f = N*torch.sum((mus_f-x_f.expand(mus_f.shape))**2 + sigmas_f**2)
    
    print_(f'mus: {mus}', mus.shape)
    print_(f'mus_i: {mus_i}', mus_i.shape)
    print_(f'mus_f: {mus_f}', mus_f.shape)
    
    # Loss_KL
    """
    paths = []
    paths_cond_probs = []
    for _ in range(M_MC):
        path, path_cond_probs = model.sample(gammas, mus, sigmas)
        # path size = [N]
        # path_cond_prob size = [N]
        paths.append(path)
        paths_cond_probs.append(path_cond_probs)
    paths = torch.stack(paths) # size [M, N]
    paths_cond_probs = torch.stack(paths_cond_probs) # size [M, N]
    """
    #paths_sample, _ = model.sample_tocho(M_MC, gammas, mus, sigmas)
    paths, paths_cond_probs = model.sample_tocho(M_MC, gammas, mus, sigmas)
    #paths = paths.unsqueeze(0)
    #paths_cond_probs = paths_cond_probs.unsqueeze(0)
    print_(f'stacked paths: {paths}', paths.shape)
    print_(f'stacked probs: {paths_cond_probs}', paths_cond_probs.shape)
    
    # We compute the actions of all the paths
    L_paths = potential(paths, h, m, w) # lagrangiangs of all paths, size [M, N-1]
    S_paths = h * torch.sum(L_paths, dim=1) # actions of all paths, size [M]
    print_(f'L_paths: {L_paths}', L_paths.shape)
    print_(f'S_paths: {S_paths}', S_paths.shape)
    
    M_MC = torch.tensor(M_MC)
    log_q_phi = torch.sum(torch.log(paths_cond_probs), dim=1)
    f = S_paths + hbar*log_q_phi
    loss_KL = (1/M_MC)*torch.sum(f)
    MC_err = torch.sqrt((1/M_MC)*torch.sum(f**2)-loss_KL**2) / torch.sqrt(M_MC)

    print_(f'log_q_phi: {log_q_phi}', log_q_phi.shape)
    print_(f'f: {f}', f.shape)
    
    print_(f'loss_KL: {loss_KL}', loss_KL.shape)
    print_(f'loss_i: {loss_i}')
    print_(f'loss_f: {loss_f}')

    # Total loss
    loss = loss_KL + loss_i + loss_f

    return loss, loss_KL, loss_i, loss_f, MC_err, paths, log_q_phi, S_paths