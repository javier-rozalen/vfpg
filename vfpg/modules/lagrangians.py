# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 00:12:37 2022

@author: javir
"""

def L_HO(paths, h, m, w):
    """
    1D, 1 particle H.O. Euclidean space Lagrangian (equivalent to the Hamiltonian).

    Parameters
    ----------
    paths : tensor
        all paths, size [M,N].
    h : tensor
        time step.
    w : float
        angular frequency.
    m : float
        mass of the particle.
    
    Returns
    -------
    L : tensor
        tensor with the lagrangian at each path position.

    """

    K = ((paths[:,1:] - paths[:,:-1]) / h)**2
    V = (w*(paths[:,1:] + paths[:,:-1]) / 2)**2
    L = 0.5 * m * (K + V)
    
    return L

####################### TESTS #######################
if __name__ == '__main__':  
    pass