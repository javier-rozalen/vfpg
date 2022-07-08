import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

def loss_paths_plot(bound, path_manifold, current_epoch, 
                    loss_list, delta_L):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 10))
    
    # Loss
    ax0 = ax[0]
    ax0.set_title('Loss', fontsize=17)
    ax0.set_xlabel('Epoch', fontsize=17)
    ax0.tick_params(axis='both', labelsize=15)
    x_axis = [i for i in range(current_epoch + 1)]
    ax0.plot(x_axis, loss_list, 
             label='$\mathcal{L}$, $\delta\mathcal{L}=$ '
             +f'{round(delta_L.item(), 2)}')
    ax0.legend(fontsize=16)
    
    # Paths
    ax1 = ax[1]
    ax1.set_title(r'Path $x(\tau)$', fontsize=17)
    ax1.set_ylabel(r'x', fontsize=17)
    ax1.tick_params(axis='both', labelsize=15)
    for path in path_manifold[:bound]:
        ax1.plot(path.detach())
        
    plt.show()
    
def metropolis_histo():
    pass