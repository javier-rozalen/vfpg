# -*- coding: utf-8 -*-
######################## IMPORTS ########################
import numpy as np

######################## AUXILIARY FUNCTION ########################
def histogram(x,dx):
    """
    Counts the frequency of appearence of points in a 100-point grid.

    Parameters
    ----------
    x : list/numpy array
        Path.
    dx : float
        Grid scale.

    Returns
    -------
    numpy array
        1D position grid with N=100, dx=dx.

    """
    count = [0]*100
    n = len(x)
	
    for i in range(n):
		
        j = 0
        done = False
        while -5 + j*dx <= +5 and done == False:
			
            if x[i] >= -5 + j*dx and x[i] <= -5 + (j + 1)*dx:
                try:
                    count[j] += 1
                except:
                    pass
                done = True
            else:
                j += 1
    return np.array(count)