######### IMPORTS ##########
import torch,math
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
###########################

############################ AUXILIARY FUNCTIONS ##############################
def loss_neg_log_like(model,train_data,target_data):
    """
    Negative log-ligelihood (MDNs). 

    Parameters
    ----------
    train_data : tensor
        Training set.

    Returns
    -------
    Loss.

    """
    
    output = model(target_data,train_data)
    q_tensor,params_gamma,params_mu,params_sigma = output[0],output[1],output[2],output[3]
    I = -torch.sum(torch.log(q_tensor))
    
    return I,params_gamma,params_mu,params_sigma

def loss_MSE(model,train_data,target_data):
    """
    Mean Squared Error.

    Parameters
    ----------
    train_data : tensor
        training data.

    Returns
    -------
    Loss.

    """

    # print(train_data.size(),train_data.dim())
    output = model(target_data,train_data)
    q_tensor,params_gamma,params_mu,params_sigma = output[0],output[1],output[2],output[3]

    # MSE
    M = len(train_data)
    I = nn.MSELoss()
    I = I(q_tensor,target_data[:M])

    return [I]

def loss_DKL(model,train_data,target_data):
    """
    Kullback-Leibler divergence.

    Parameters  
    ----------
    train_data : tensor
        training data.

    Returns
    -------
    Loss.

    """
    
    output = model(target_data,train_data)
    q_tensor,params_gamma,params_mu,params_sigma = output[0],output[1],output[2],output[3]

    # Monte Carlo
    M = len(train_data)
    I = (1/M)*torch.sum(torch.log(target_data)-torch.log(q_tensor))
    I2 = (1/M)*torch.sum((torch.log(target_data)-torch.log(q_tensor))**2)
    error = (1/math.sqrt(M))*torch.sqrt(I2-I**2)

    return I,params_gamma,params_mu,params_sigma

    
def loss_MSE_simple(model,train_data,target_data):
    """
    Mean Squared Error.

    Parameters
    ----------
    train_data : tensor
        training data.

    Returns
    -------
    Loss.

    """

    # print(train_data.size(),train_data.dim())
    q_tensor = model(train_data)

    # MSE
    M = len(train_data)
    I = nn.MSELoss()
    I = I(q_tensor,target_data[:M])

    return [I]
    
def loss_DKL_simple(model,train_data,target_data):
    """
    Kullback-Leibler divergence.

    Parameters  
    ----------
    train_data : tensor
        training data.

    Returns
    -------
    Loss.

    """
    
    q_tensor = model(train_data)
    #print(q_tensor)

    # Monte Carlo
    M = len(train_data)
    I = (1/M)*torch.sum(torch.log(target_data)-torch.log(q_tensor))
    I2 = (1/M)*torch.sum((torch.log(target_data)-torch.log(q_tensor))**2)
    error = (1/math.sqrt(M))*torch.sqrt(I2-I**2)
    #print(I)

    return I.unsqueeze(0)
    
    
    
    
    