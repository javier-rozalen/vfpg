import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

# My modules
from modules.physical_constants import hbar

def loss_plot(loss_list, MC_error, current_epoch, save=False, plot_path=''):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 8))
    
    # Loss
    ax.set_title('Loss', fontsize=17)
    ax.set_xlabel('Epoch', fontsize=17)
    ax.tick_params(axis='both', labelsize=15)
    x_axis = [i for i in range(current_epoch + 1)]
    y_axis = [e.cpu().detach() for e in loss_list]
    MC_error = MC_error.cpu().detach()
    ax.plot(x_axis, y_axis, 
             label='$\mathcal{L}$, $\delta\mathcal{L}=$ '
             +f'{round(MC_error.item(), 2)}')
    ax.legend(fontsize=16)
        
    # Save
    if save:
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        print(f'Plot correctly saved at: {plot_path}')
    
    plt.show()
    
def histogram_plot(x_axis, y_axis, path_manifold, bound, time_grid,
                   save_plot=False, plot_path=''):
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    
    # Wave function
    ax = axes[0]
    sigma = np.sqrt(1.)
    x_target = np.linspace(-3, 3, 200)
    y_target = (((1/(np.pi*sigma**2))**(1/4))*np.exp(-x_target**2/(2*sigma**2)))**2
    ax.hist(x_axis, weights=y_axis, bins=int(len(x_axis)))
    #ax.plot(x_axis,y_axis,linestyle='none',marker='o')
    ax.plot(x_target, y_target, label='$|\Psi_0(x)|^2,\quad\sigma=1$')
    ax.set_xlim(-5, 5)
    ax.set_ylabel('$|\Psi(x)|^2$', fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(fontsize=16)
    
    # Paths
    ax = axes[1]
    ax.set_title(r'Paths $x(\tau)$', fontsize=17)
    ax.set_xlabel('$x$', fontsize=17)
    ax.set_ylabel(r'$\tau$', fontsize=17)
    ax.tick_params(axis='both', labelsize=15, labeltop=True, top=True)
    for path in path_manifold[:bound]:
        ax.plot(path.detach(), time_grid)
    
    # Save
    if save_plot:
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        print(f'Plot correctly saved at: {plot_path}')
    
    plt.show()
    
