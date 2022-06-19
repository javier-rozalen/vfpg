import torch

def train_loop(model, loss_fn, optimizer, train_set=0, target_data=0, h=0):
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
    loss_output, MC_error, paths = loss_fn(model, train_set, target_data, h)
    loss_output.backward()
    optimizer.step()
    return loss_output, MC_error, paths
    
####################### TESTS #######################
if __name__ == '__main__':  
    pass