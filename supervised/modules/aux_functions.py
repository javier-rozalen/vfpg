import torch

def train_loop(model,train_set,target_data,loss_fn,optimizer):
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
    loss_output = loss_fn(model,train_set,target_data)
    loss_output[0].backward()
    optimizer.step()
    return loss_output
    
####################### TESTS #######################
if __name__ == '__main__':  
    pass