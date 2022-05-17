import numpy as np
from tqdm import tqdm
import torch,math

# My modules
from modules.neural_networks import q_phi
from modules.plotters import loss_plot
from modules.actions import S_HO
from modules.integration import simpson_weights

# General parameters
Nhid = 50
n_epochs = 100
N = 50
M = 1000
hbar = 1.
m = 1.
w = 1.

# Training parameters
learning_rate = 1e-2
epsilon = 1e-8
smoothing_constant = 0.9

path_manifold = [torch.tensor([1.]*N) for _ in range(M)]
x_i = 0.
x_f = 0.
t_0 = 0.
t_f = 1.
t = [torch.tensor(e) for e in np.linspace(t_0,t_f,N)]
h = t[1]-t[0]
int_weights = simpson_weights(t)

def gaussian_mixture(x,params):
    
    return torch.sum(params)

def loss():
    """
    Loss function.

    Returns
    -------
    loss : tensor
        Kullback-Leibler divergence.

    """
    global I,I2,error
    
    q_manifold,S_manifold = [],[]
    for x in path_manifold:
        q_params = q_params_nn(x)
        q_x = gaussian_mixture(x,q_params)
        S_x = S_HO(x,h,m,w)

        # Appends
        q_manifold.append(q_x)
        S_manifold.append(S_x)
        
    q_tensor = torch.stack(q_manifold)
    S_tensor = torch.stack(S_manifold)
    
    # Monte Carlo integration 
    I = (1/M)*torch.sum(torch.log(q_tensor)+(1/hbar)*S_tensor)
    I2 = (1/M)*torch.sum((torch.log(q_tensor)+(1/hbar)*S_tensor)**2)
    error = (1/math.sqrt(M))*torch.sqrt(I2-I**2)
    
    return I

Nin = N
Nout = Nin
W1 = torch.rand(Nhid,Nin,requires_grad=True)*(-1.) 
B = torch.rand(Nhid)*2.-torch.tensor(1.) 
W2 = torch.rand(Nout,Nhid,requires_grad=True) 

q_params_nn = q_phi(Nin,Nhid,Nout,W1,W2,B)

loss_fn = loss
optimizer = torch.optim.RMSprop(params=q_params_nn.parameters(),lr=learning_rate,eps=epsilon)

def train_loop(loss_fn,optimizer):
    """
    Training loop.

    Parameters
    ----------
    loss_fn : function
        loss function.
    optimizer : torch.optim
        optimizer.

    Returns
    -------
    None.

    """  
    optimizer.zero_grad()
    loss_fn().backward()
    optimizer.step()

# Epoch loop
loss_list = []
x_axis = []
for t in tqdm(range(n_epochs)):
    train_loop(loss_fn,optimizer)
    loss_list.append(I.item())
    x_axis.append(t)
    loss_plot(x=x_axis,y=loss_list)