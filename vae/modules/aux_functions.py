import torch, os
import numpy as np

def train_loop(dev, model, loss_fn, optimizer, train_set=0, target_data=0):
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
    Loss.

    """  
    optimizer.zero_grad()
    loss, MC_error = loss_fn(model, train_set, dev)
    loss.backward()
    optimizer.step()
    
    return loss, MC_error

def show_layers(model):
    print("\nLayers and parameters:\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:100]} \n")
        
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def dir_support(list_of_nested_dirs):
    """
    Directories support: ensures that the (nested) directories given via the 
    input list do exist, creating them if necessary. 
    
    Parameters
    ----------
    nested_dirs : list
        Contains all nested directories in order.
    Returns
    -------
    None.
    """
    for i in range(len(list_of_nested_dirs)):
        potential_dir = '/'.join(list_of_nested_dirs[:i+1]) 
        if not os.path.exists(potential_dir):
            os.makedirs(potential_dir)
            print(f'Creating directory {potential_dir}...')

def fetch_data(n_examples, paths_file, actions_file):
    path_manifold, S_manifold = [], []
    with open(paths_file, 'r') as file:
        for line in file.readlines():
            path = [torch.tensor(float(x)) for x in line.split(' ')]
            path_manifold.append(torch.stack(path))
        file.close()
    with open(actions_file, 'r') as file:
        for line in file.readlines():
            S = torch.tensor(float(line))
            S_manifold.append(S)
        file.close()
    x_tensor = torch.stack(path_manifold)
    S_tensor = torch.stack(S_manifold)

    train_set = x_tensor[:n_examples]
    actions_set = S_tensor[:n_examples]
    
    return train_set, actions_set 

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
        K = ((x_i1-x_i)/h)**2
        V = (w*(x_i1+x_i)/2)**2
        S_prime += K + V
        
    return 0.5*m*h*S_prime
    
####################### TESTS #######################
if __name__ == '__main__':  
    pass