import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

def pic_creator(bound,j):
    """
    Plots the paths in path_manifold present at current execution time as well as 
    the loss funciton.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(14,6))
    
    ax0 = ax[0]
    ax0.set_title(r'Path $x(\tau)$',fontsize=17)
    ax0.set_xlabel(r'$\tau$',fontsize=17)
    ax0.set_ylabel('$x$',rotation=180,labelpad=10,fontsize=17)
    ax0.tick_params(axis='both',labelsize=15)
    c = 0
    for path,action in zip(path_manifold,S_manifold):
        if c<bound:
            ax0.plot([e.item() for e in t],[e.item() for e in path],label=f'S = {round(action.item(),3)}')
        c+=1    
        
    #ax0.legend(loc='center',bbox_to_anchor=(0.5,-0.35))
    
    ax1 = ax[1]
    ax1.set_title('Loss',fontsize=17)
    ax1.set_xlabel('Epoch',fontsize=17)
    ax1.tick_params(axis='both',labelsize=15)
    ax1.plot([e for e in range(j+1)],loss_list,label='$\mathcal{L}$')
    ax1.plot([e for e in range(j+1)],loss_kl_list,label='$\mathcal{L}_{KL}-lnZ$')
    ax1.plot([e for e in range(j+1)],loss_i_list,label='$\mathcal{L}_i$')
    ax1.plot([e for e in range(j+1)],loss_f_list,label='$\mathcal{L}_f$')    
    ax1.axhline(0.,color='red',linestyle='dashed')
    ax1.legend(fontsize=16)
    
    plt.show()