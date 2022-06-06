import matplotlib.pyplot as plt
import numpy as np

def loss_plot(x,y,mesh,mu,sigma,loss,adaptive_factor,mc_error,color,save=False,model_name=''):
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(10,6))
    
    ax = axes[0]
    ax.set_xlabel('Epochs',fontsize=16)
    ax.set_ylabel('Loss',fontsize=16)
    if adaptive_factor != 0:
        ax.set_ylim(loss-abs(loss*adaptive_factor),loss+abs(loss*adaptive_factor))
    ax.set_title('Loss function',fontsize=18)
    ax.tick_params(axis='both',which='both',labelsize=15)
    ax.plot(x,y,color=color,label=f'$\sigma=$ {mc_error:.3f}')
    ax.legend(fontsize=14)
    
    ax = axes[1]
    ax.set_xlabel('$x_i$',fontsize=16)
    ax.set_ylabel('$\sigma_i$, $\mu_i$',fontsize=16)
    ax.set_title('Parameters',fontsize=18)
    ax.tick_params(axis='both',which='both',labelsize=15)
    ax.plot(mesh[:100].detach(),mu[:100].detach(),'bo',markersize=1)
    ax.plot(mesh[:100].detach(),sigma[:100].detach(),'ro',markersize=1)
    #ax.legend(fontsize=14)
            
    if save == True:
        plt.savefig(model_name)    
    plt.pause(0.001)
    
def loss_plot_rnn(x,y,loss,adaptive_factor,mc_error,color,save=False,model_name=''):
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,6))
    
    ax.set_xlabel('Epochs',fontsize=16)
    ax.set_ylabel('Loss',fontsize=16)
    if adaptive_factor != 0:
        ax.set_ylim(loss-abs(loss*adaptive_factor),loss+abs(loss*adaptive_factor))
    ax.set_title('Loss function',fontsize=18)
    ax.tick_params(axis='both',which='both',labelsize=15)
    ax.plot(x,y,color=color,label=f'$\sigma=$ {mc_error:.3f} %')
    ax.legend(fontsize=14)
            
    if save == True:
        plt.savefig(model_name)    
    plt.pause(0.001)

def histo2(x_axis,y_axis,S_paths,mu,sigma,n_accepted,path,Nhid,num_layers,learning_rate):
    fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(10,8))
    
    # Wave function
    ax = axes[0][0]
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
    ax.legend(fontsize=15)
    
    # Paths
    ax = axes[1][0]
    ax.set_title('Last accepted path',fontsize=17)
    ax.set_xlabel(r'x',fontsize=17)
    ax.set_ylabel('t',labelpad=10,fontsize=17)
    ax.set_xlim(-5,5)
    ax.tick_params(axis='both',labelsize=15)
    t_grid = np.linspace(0,100,len(path))
    ax.plot(path,t_grid)
    for x in x_axis:
        ax.axvline(x)
    """
    # Gaussians
    ax = axes[0][1]
    ax.set_title('$q_{\phi}$',fontsize=17)
    ax.set_xlabel(r'x',fontsize=17)
    ax.set_ylabel('gaussians',fontsize=17)
    ax.set_xlim(-5,5)
    ax.tick_params(axis='both',labelsize=15)
    x = np.linspace(-5,5,50)
    mu = mu.numpy()[0]
    #sigma = sigma.numpy()
    ax.plot(x,(1/(np.sqrt(2*np.pi)*sigma))*np.exp(-0.5*((x-mu)/sigma)**2))
    """
    plt.show()