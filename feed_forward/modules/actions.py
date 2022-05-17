# -*- coding: utf-8 -*-
"""
Created on Tue May 10 09:40:48 2022

@author: javir
"""
import numpy as np

def S_HO(x,h,m,w):
    """
    Euclidean-time action of the 1D, 1-particle H.O.
    
    Parameters
    ----------
    x : list
        (positions of the) Path.

    Returns
    -------
    S : float
        Action of the path given as input.

    """
    S_prime = 0.
    for i in range(len(x)-1):
        x_i1 = x[i+1]
        x_i = x[i]
        S_prime += ((x_i1-x_i)/h)**2+(w*(x_i1+x_i)/2)**2
        
    return 0.5*m*h*S_prime