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
    
def histogram_plot(x_axis, y_axis, y_MCMC, path_manifold, bound, time_grid,
                   save_plot=False, plot_path=''):
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    
    # Wave function
    ax = axes[0]
    ax.hist(x_axis, weights=y_axis, bins=int(len(x_axis)))
    ax.plot(x_axis, y_MCMC, label='MCMC fit')
    ax.set_xlim(-3, 3)
    ax.set_xlabel('$x$', fontsize=15)
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
        ax.plot(path.cpu().detach(), time_grid)
    
    # Save
    if save_plot:
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        print(f'Plot correctly saved at: {plot_path}')
    
    plt.show()
    
def master_plot(x_axis_train, y_axis_train, x_axis_test, y_axis_test,
                save_plot=False, plot_path=''):
    
    """Plots logp(x) vs S(x)/hbar"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
       
    ax.set_title(r'Path distribution', fontsize=17)
    ax.set_xlabel(r'$S_E(\vec{x})/\hbar$', fontsize=17)
    ax.set_ylabel(r'$\log p(\vec{x})$', fontsize=17)
    ax.tick_params(axis='both', labelsize=15, top=True)
    ax.set_xlim(0, 20)
    ax.set_ylim(-20, 0)
    
    ax.plot(x_axis_train, [-e for e in x_axis_train], color='red', label='Exact')
    ax.scatter(x_axis_train, y_axis_train, color='blue', label='Train') # Train data
    ax.scatter(x_axis_test, y_axis_test, facecolors='none', edgecolors='orange', label='Test') # Test data
    ax.legend(fontsize=15)
    
    # Save
    if save_plot:
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        print(f'Plot correctly saved at: {plot_path}')
    
    plt.show()
    
