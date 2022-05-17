# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 00:12:37 2022

@author: javir
"""

import torch

def L_HO(x,t,m,w):
    """
    1D, 1 particle H.O. Euclidean space Lagrangian (equivalent to the Hamiltonian).

    Parameters
    ----------
    x : list
        positions of path.
    t : list
        time steps of path.
    w : float
        angular frequency.
    m : float
        mass of the particle.
    
    Returns
    -------
    L : tensor
        tensor with the lagrangian at each path position.

    """

    L = []
    h = t[1]-t[0]
    w2 = w**2
    for i in range(len(t)):
        if i == len(t)-1:
            L.append(((x[i]-x[i-1])/h)**2+w2*(((x[i]+x[i-1])/2)**2))
        else:
            L.append(((x[i+1]-x[i])/h)**2+w2*(((x[i+1]+x[i])/2)**2))
            
    L = 0.5*m*torch.cat(L)
    return L

def L_fp(x,t,m,w):
    """
    1D, free particle Lagrangian.

    Parameters
    ----------
    x : list
        positions of path.
    t : list
        time steps of path.
    m : float
        mass of the particle.
    
    Returns
    -------
    L : tensor
        tensor with the lagrangian at each path position.

    """

    L = []
    h = t[1]-t[0]
    for i in range(len(t)-1):
        L.append(((x[i+1]-x[i])/h)**2)
            
    L = 0.5*m*torch.cat(L)
    return L

####################### TESTS #######################
if __name__ == '__main__':  
    pass