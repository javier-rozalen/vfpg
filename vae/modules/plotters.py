import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

# My modules
from modules.physical_constants import hbar

def loss(loss_list, delta_L, current_epoch):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    
    # Loss
    ax.set_title('Loss', fontsize=17)
    ax.set_xlabel('Epoch', fontsize=17)
    ax.tick_params(axis='both', labelsize=15)
    x_axis = [i for i in range(current_epoch + 1)]
    ax.plot(x_axis, loss_list, 
             label='$\mathcal{L}$, $\delta\mathcal{L}=$ '
             +f'{round(delta_L.item(), 2)}')
    ax.legend(fontsize=16)
        
    plt.show()
    
