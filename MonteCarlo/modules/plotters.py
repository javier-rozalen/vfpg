# -*- coding: utf-8 -*-
######################## IMPORTS ########################
import matplotlib.pyplot as plt
import numpy as np

######################## PLOTTING FUNCTIONS ########################
def nice_plot(x_axis, y_axis, S_paths, n_accepted, path, plot_name='', 
              save=False):
    fig, axes = plt.subplot_mosaic([['histo', 'action'], ['path', 'action']], 
                                   figsize=(14, 10))
    
    # Wave function
    ax = axes['histo']
    sigma = np.sqrt(1.)
    x_target = np.linspace(-3, 3, 200)
    y_target = ((1/(np.pi*sigma**2))**(1/4))*np.exp(-x_target**2/(2*sigma**2))
    y_target = y_target**2
    ax.hist(x_axis, weights=y_axis, bins=int(len(x_axis)))
    ax.plot(x_target, y_target, label='Exact')
    ax.set_xlim(-3, 3)
    ax.set_ylabel('$|\Psi(x)|^2$', fontsize=17)
    ax.tick_params(axis='both', labelsize=17)
    ax.legend(fontsize=17)
    
    # Action
    ax = axes['action']
    ax.set_title(r'$S_E[x(\tau)]$', fontsize=18)
    ax.set_xlabel(r'$N_{paths}$', fontsize=17)
    ax.tick_params(axis='both', labelsize=17)
    ax.plot([i for i in range(n_accepted)], S_paths)
    
    # Path
    ax = axes['path']
    ax.set_xlabel(r'x', fontsize=17)
    ax.set_ylabel('t', labelpad=5, fontsize=17)
    ax.set_xlim(-3, 3)
    ax.tick_params(axis='both', labelsize=17)
    t_grid = np.linspace(0, 100, len(path))
    ax.plot(path, t_grid)
    for x in x_axis:
        ax.axvline(x, alpha=0.4)
    if save:
        plt.savefig(plot_name)
        print(f'Plot saved at {plot_name}.')
    plt.show()
        