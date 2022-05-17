# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 00:06:52 2022

@author: javir
"""
import torch

def simpson_weights(x):
    """
    Returns the weights used for integration in the 1D Simpson method.

    Parameters
    ----------
    x : list
        Domain of integration.

    Returns
    -------
    tensor (32-bit)
        Weights of integration.

    """
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
    
    return torch.tensor(w_k,dtype=torch.float32)

####################### TESTS #######################
if __name__ == '__main__':  
    pass