import torch
import numpy as np

def train_loop(model, train_set, loss_fn, optimizer, mus, sigmas):
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
    loss_output, MC_error, paths = loss_fn(model=model, 
                                           train_set=train_set,
                                           mus=mus,
                                           sigmas=sigmas)
    loss_output.backward()
    optimizer.step()
    return loss_output, MC_error, paths

def show_layers(model):
    print("\nLayers and parameters:\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:100]} \n")
        
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
####################### TESTS #######################
if __name__ == '__main__':  
    pass