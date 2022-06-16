# -*- coding: utf-8 -*-
"""
Created on Tue May 10 09:42:50 2022

@author: javir
"""
import matplotlib.pyplot as plt
import numpy as np

def S_vs_npaths(paths,S_paths):
    """
    Plots the evolution of the action against the number of accepted paths.
    
    Returns
    -------
    None.

    """
    
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(14,6))
    
    ax.set_title(r'S_E[$x(\tau)$]',fontsize=17)
    ax.set_xlabel(r'N_paths',fontsize=17)
    ax.set_ylabel('S',rotation=180,labelpad=10,fontsize=17)
    ax.tick_params(axis='both',labelsize=15)
    
    ax.plot([i for i in range(len(paths))],S_paths)

    plt.show()
    plt.pause(0.001)
    
def histo(paths,paths_sym,S_paths,M,N,seed,plot_name='',save=False):
    """
    Plots a histogram of the input list "paths".

    Parameters
    ----------
    paths : numpy array or list
        list containing all the paths.

    Returns
    -------
    None.

    """
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(14,6))
    
    # Histogram + targets
    ax = axes[0]
    x_target = np.linspace(-6,6,200)
    sigma = np.sqrt(1.)
    sigma2 = np.sqrt(2.)
    y_target = (((1/(np.pi*sigma**2))**(1/4))*np.exp(-x_target**2/(2*sigma**2)))**2
    y_target2 = (((1/(np.pi*sigma2**2))**(1/4))*np.exp(-x_target**2/(2*sigma2**2)))**2
    ax.set_ylabel('$|\Psi(x)|^2$')
    ax.set_xlabel('x')
    ax.set_xlim(-3,3)
    ax.set_title(f'M = {M}, N = {N+1}, n_paths = {len(paths)}, Seed = {seed}')
    ax.tick_params(axis='both',labelsize=15)
    
    ax.hist(paths_sym, density=True, bins=100)  # density=False would make counts
    ax.plot(x_target,y_target,label='$|\Psi_0(x)|^2,\quad\sigma=1$')
    ax.plot(x_target,y_target2,label='$|\Psi_0(x)|^2,\quad\sigma=\sqrt{2}$')
    ax.legend()
    
    # Action
    ax = axes[1]
    ax.set_title(r'S_E[$x(\tau)$]',fontsize=17)
    ax.set_xlabel(r'N_paths',fontsize=17)
    ax.set_ylabel('S',rotation=180,labelpad=10,fontsize=17)
    ax.tick_params(axis='both',labelsize=15)
    
    #ax.plot([i for i in range(len(paths_sym))],[S_paths[i//2] for i in range(len(S_paths)*2)])
    
    if save:
        plt.savefig(plot_name)
        print(f'Plot saved {plot_name}.')
    plt.show()
    
def histo2(x_axis,y_axis,S_paths,n_accepted,path,plot_name='',save=False):
    fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(14,10))
    
    # Wave function
    ax = axes[0][0]
    sigma = np.sqrt(1.)
    x_target = np.linspace(-3,3,200)
    y_target = (((1/(np.pi*sigma**2))**(1/4))*np.exp(-x_target**2/(2*sigma**2)))**2
    ax.hist(x_axis,weights=y_axis,bins=int(len(x_axis)))
    #ax.plot(x_axis,y_axis,linestyle='none',marker='o')
    ax.plot(x_target,y_target,label='$|\Psi_0(x)|^2,\quad\sigma=1$')
    ax.set_xlim(-3,3)
    ax.set_ylabel('$|\Psi(x)|^2$',fontsize=15)
    ax.tick_params(axis='both',labelsize=15)
    ax.legend()
    
    # Action
    ax = axes[0][1]
    ax.set_title(r'S_E[$x(\tau)$]',fontsize=17)
    ax.set_xlabel(r'N_paths',fontsize=17)
    ax.set_ylabel('S',rotation=180,labelpad=10,fontsize=17)
    ax.tick_params(axis='both',labelsize=15)
    ax.plot([i for i in range(n_accepted)],S_paths)
    
    # Paths
    ax = axes[1][0]
    ax.set_title('Last accepted path',fontsize=17)
    ax.set_xlabel(r'x',fontsize=17)
    ax.set_ylabel('t',labelpad=10,fontsize=17)
    ax.set_xlim(-3,3)
    ax.tick_params(axis='both',labelsize=15)
    t_grid = np.linspace(0,100,len(path))
    ax.plot(path,t_grid)
    for x in x_axis:
        ax.axvline(x)
        
    if save:
        plt.savefig(plot_name)
        print(f'Plot saved {plot_name}.')
    plt.show()
        