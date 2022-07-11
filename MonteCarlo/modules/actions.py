# -*- coding: utf-8 -*-

######################## ACTIONS ########################
def S_HO(x, h, m, w):
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

def S_double_well(x, h, m, w):
    S_prime = 0.
    alpha = 0.05
    beta = -1.
    for i in range(len(x)-1):
        x_i1 = x[i+1]
        x_i = x[i]
        K = ((x_i1-x_i)/h)**2
        V = alpha*((x_i1+x_i)/2)**4 + beta*((x_i1+x_i)/2)**2
        S_prime +=  K + V
        
    return 0.5*m*h*S_prime