import matplotlib.pyplot as plt

def loss_plot(x,y):
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6))
    
    ax.set_xlabel('Epochs',fontsize=16)
    ax.set_ylabel('Loss',fontsize=16)
    ax.set_title('Loss function',fontsize=18)
    ax.tick_params(axis='both',which='both',labelsize=15)
    ax.plot(x,y)
    
    plt.pause(0.001)