import matplotlib.pyplot as plt
import numpy as np

def loss_plot(x,y,loss,adaptive_factor,color,save=False,model_name=''):
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6))
    
    ax.set_xlabel('Epochs',fontsize=16)
    ax.set_ylabel('Loss',fontsize=16)
    if adaptive_factor != 0:
        ax.set_ylim(loss-abs(loss*adaptive_factor),loss+abs(loss*adaptive_factor))
    ax.set_title('Loss function',fontsize=18)
    ax.tick_params(axis='both',which='both',labelsize=15)
    ax.axhline(0.,linestyle='--',color='red')
    ax.plot(x,y,color=color)
    
    if save == True:
        plt.savefig(model_name)    
    plt.pause(0.001)

    
def histo2(x_axis,y_axis,S_paths,n_accepted,path,Nhid,num_layers,learning_rate):
    fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(10,12))
    
    # Wave function
    ax = axes[0]
    sigma = np.sqrt(1.)
    x_target = np.linspace(-3,3,200)
    y_target = (((1/(np.pi*sigma**2))**(1/4))*np.exp(-x_target**2/(2*sigma**2)))**2
    ax.hist(x_axis,weights=y_axis,bins=int(len(x_axis)))
    #ax.plot(x_axis,y_axis,linestyle='none',marker='o')
    ax.plot(x_target,y_target,label='$|\Psi_0(x)|^2,\quad\sigma=1$')
    ax.set_title(f'Nhid : {Nhid}, # layers : {num_layers}, lr : {learning_rate}',fontsize=18)
    ax.set_xlim(-5,5)
    ax.set_ylabel('$|\Psi(x)|^2$',fontsize=15)
    ax.tick_params(axis='both',labelsize=15)
    ax.legend()
    
    # Paths
    ax = axes[1]
    ax.set_title('Last accepted path',fontsize=17)
    ax.set_xlabel(r'x',fontsize=17)
    ax.set_ylabel('t',labelpad=10,fontsize=17)
    ax.set_xlim(-5,5)
    ax.tick_params(axis='both',labelsize=15)
    t_grid = np.linspace(0,100,len(path))
    ax.plot(path,t_grid)
    for x in x_axis:
        ax.axvline(x)
    
    plt.show()