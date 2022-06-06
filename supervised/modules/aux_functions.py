import torch

def train_loop(train_data,loss_fn,optimizer):
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
    loss_fn(train_data).backward()
    optimizer.step()
    
####################### TESTS #######################
if __name__ == '__main__':  
    pass