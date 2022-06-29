import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

# My modules
from modules.physical_constants import hbar

def loss_paths_plot_ours(bound, time_grid, path_manifold, wf, current_epoch, 
                    loss_list, loss_KL_list, loss_i_list, loss_f_list, 
                    delta_L, show_loss_i=False, show_loss_f=False):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    
    # Loss
    ax0 = ax[0][0]
    ax0.set_title('Loss', fontsize=17)
    ax0.set_xlabel('Epoch', fontsize=17)
    ax0.tick_params(axis='both', labelsize=15)
    x_axis = [i for i in range(current_epoch + 1)]
    ax0.plot(x_axis, loss_list, 
             label='$\mathcal{L}$, $\delta\mathcal{L}=$ '
             +f'{round(delta_L.item(), 2)}')
    if show_loss_i:
        ax0.plot(x_axis, loss_i_list, label='$\mathcal{L}_i$')
    if show_loss_f:
        ax0.plot(x_axis, loss_f_list, label='$\mathcal{L}_f$')
    ax0.legend(fontsize=16)
    
    # Wave functions
    ax1 = ax[0][1]
    x_axis_hist = np.linspace(-4.95, 4.95, 100)
    sigma = np.sqrt(1.)
    x_target = np.linspace(-3, 3, 200)
    y_target = (((1/(np.pi*sigma**2))**(1/4))*np.exp(-x_target**2/(2*sigma**2)))**2
    ax1.hist(x_axis_hist, weights=wf, bins=int(len(x_axis_hist)))
    ax1.plot(x_target,y_target,label='$|\Psi_0(x)|^2,\quad\sigma=1$')
    ax1.set_xlim(-3, 3)
    ax1.set_ylabel('$|\Psi(x)|^2$', fontsize=17)
    ax1.tick_params(axis='both', labelsize=15)
    ax1.legend(fontsize=16)
    
    # Paths
    ax3 = ax[1][1]
    ax3.set_title(r'Path $x(\tau)$', fontsize=17)
    ax3.set_xlabel('$x$', fontsize=17)
    ax3.set_ylabel(r'$\tau$', fontsize=17)
    #ax3.set_xlim(-4, 4)
    ax3.tick_params(axis='both', labelsize=15)
    for path in path_manifold[:bound]:
        ax3.plot(path.detach(), time_grid)
        
    plt.show()
    
def loss_paths_plot_theirs(bound, time_grid, path_manifold, wf, current_epoch, 
                    loss_list, loss_KL_list, loss_i_list, loss_f_list, 
                    delta_L, show_loss_i=False, show_loss_f=False, 
                    log_q_phi=0., S_paths=0.):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
    fig.subplots_adjust(hspace=0.4)
    
    # ln(q) vs. S_E
    ax0 = ax[0][0]
    ax0.set_title('Path distribution', fontsize=17)
    ax0.set_xlabel('$(S_E-S_{E_{min}})/\hbar$', fontsize=17)
    ax0.set_ylabel('$\ln q + b$', fontsize=17)
    ax0.tick_params(axis='both', labelsize=15)
    S_E_min = min(S_paths)
    x_axis = (S_paths - S_E_min.repeat(S_paths.size(0))) / hbar
    b = (-loss_KL_list[-1] + S_E_min) / hbar
    y_axis = log_q_phi + b.repeat(log_q_phi.size(0))
    ax0.scatter(x_axis.detach(), y_axis.detach())
    #ax0.plot(x_axis.detach(), -x_axis.detach(), color='red', label='Exact')
    ax0.legend(fontsize=16)
    
    # Loss
    ax1 = ax[0][1]
    ax1.set_title('Loss', fontsize=17)
    ax1.set_xlabel('Epoch', fontsize=17)
    ax1.tick_params(axis='both', labelsize=15)
    x_axis = [i for i in range(current_epoch + 1)]
    ax1.plot(x_axis, loss_KL_list, 
             label='$\mathcal{L}_{KL}$, $\delta\mathcal{L}_{KL}=$ '
             +f'{round(delta_L.item(), 2)}')
    if show_loss_i:
        ax1.plot(x_axis, loss_i_list, label='$\mathcal{L}_i$')
    if show_loss_f:
        ax1.plot(x_axis, loss_f_list, label='$\mathcal{L}_f$')
    ax1.legend(fontsize=16)
    
    # Paths
    ax2 = ax[1][0]
    ax2.set_title(r'Paths $x(\tau)$', fontsize=17)
    ax2.set_xlabel('$x$', fontsize=17)
    ax2.set_ylabel(r'$\tau$', fontsize=17)
    ax2.tick_params(axis='both', labelsize=15, labeltop=True, top=True)
    for path in path_manifold[:bound]:
        ax2.plot(path.detach(), time_grid)
    
    plt.show()