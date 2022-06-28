import torch
import numpy as np
from modules.lagrangians import L_HO, L_double_well

def train_loop(model, loss_fn, optimizer, train_set=0, target_data=0, 
               potential=L_HO, M_MC=100, h=0, x_i=0, x_f=1):
    """
    Training loop.

    Parameters
    ----------
    loss_fn : function
        loss function.
    optimizer : torch.optim
        optimizer.

    Returns
    -------
    None.

    """  
    optimizer.zero_grad()
    loss, loss_KL, loss_i, loss_f, MC_error, paths = loss_fn(model=model, 
                                                           train_set=train_set, 
                                                           target_set=target_data, 
                                                           potential=potential,
                                                           M_MC = M_MC,
                                                           h=h, 
                                                           x_i=x_i,
                                                           x_f=x_f)
    loss.backward()
    optimizer.step()
    return loss, loss_KL, loss_i, loss_f, MC_error, paths

def show_layers(model):
    print("\nLayers and parameters:\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:100]} \n")
        
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def histogram(x, dx):
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
        if x[i] >= -5 and x[i] <= 5:
            j = 0
            done = False
            while -5 + j*dx <= +5 and done == False:
                if x[i] >= -5 + j*dx and x[i] <= -5 + (j + 1)*dx:
                    count[j] += 1.
                    done = True
                else:
                    j += 1
    return np.array(count)
    
####################### TESTS #######################
if __name__ == '__main__':  
    pass