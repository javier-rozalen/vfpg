######### IMPORTS ##########
import torch,math
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
###########################

############################ AUXILIARY FUNCTIONS ##############################
def show_layers(model):
    print("\nLayers and parameters:\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:100]} \n")
        
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
############################## NEURAL NETWORKS ################################
class q_phi_ff_disconnected(nn.Module):
    """
    Feed-forward neural network. 
    """
    
    def __init__(self,Nin,Nhid,Nout,num_layers=1):
        super(q_phi_ff_disconnected,self).__init__()
        
        # Auxiliary stuff
        self.softplus = nn.Softplus()
        self.N = Nin
        self.actfun = nn.Sigmoid()
        
        # Initial parameters
        W1 = torch.rand(Nhid,Nin,requires_grad=True)*(-1)
        B = torch.rand(Nhid)*2-torch.tensor(1.)
        W2 = torch.rand(Nout,Nhid,requires_grad=True)
        
        # Layers
        layers_gamma,layers_mu,layers_sigma = nn.ModuleList(),nn.ModuleList(),nn.ModuleList()
        self.layer1_gamma = nn.linear(Nhid,Nout)
        self.layer1_mu = nn.Linear(Nin,Nhid)
        self.layer1_sigma = nn.Linear(Nin,Nhid)
        self.layerlast_gamma = nn.Linear(Nhid,Nout)
        self.layerlast_mu = nn.Linear(Nhid,Nout)
        self.layerlast_sigma = nn.Linear(Nhid,Nout)
        layers_gamma.append(self.layer1_gamma)
        layers_mu.append(self.layer1_mu)
        layers_sigma.append(self.layer1_sigma)
        for _ in range(num_layers - 1):
            layers_gamma.append(nn.Linear(Nhid,Nhid))
            layers_gamma.append(self.actfun)
            layers_mu.append(nn.Linear(Nhid,Nhid))
            layers_mu.append(self.actfun)
            layers_sigma.append(nn.Linear(Nhid,Nhid))
            layers_sigma.append(self.actfun)
        layers_gamma.append(self.layerlast_gamma)
        layers_mu.append(self.layerlast_mu)
        layers_sigma.append(self.layerlast_sigma)
        self.layers_gamma = layers_gamma
        self.layers_mu = layers_mu
        self.layers_sigma = layers_sigma
        
        """
        with torch.no_grad():
            self.layer1_mu.weight = nn.Parameter(W1)
            self.layer1_mu.bias = nn.Parameter(B)
            self.layer1_sigma.weight = nn.Parameter(W1)
            self.layer1_sigma.bias= nn.Parameter(B)
            self.layerlast_mu.weight = nn.Parameter(W2)
            self.layerlast_mu.bias = nn.Parameter(B)
            self.layerlast_sigma.weight = nn.Parameter(W2)
            self.layerlast_sigma.bias= nn.Parameter(B)
            """
        
    def prod_of_gaussians(self,x,params1,params2):
        global params_mu,params_sigma
        """
        Computes a pdf that is a product of gaussians given the means and stdevs.
        The return is the logarithm of the pdf. 
        
        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        params : TYPE
            DESCRIPTION.

        Returns
        -------
        tensor
            Logarithm of q_phi.

        """

        params_mu = params1
        params_sigma = self.softplus(params2)
        sum_ = 0.5*(((x-params_mu)/params_sigma)**2)+torch.log(params_sigma)
        return -(torch.sum(sum_,dim=1)+self.N/2*torch.log(torch.tensor(2*math.pi)).expand(x.size()[0]))
    
    def gaussian_mixture(self,x,params1,params2,params3):
        global params_gamma,params_mu,params_sigma
        
        params_gamma = nn.Softmax(params1)
        params_mu = params2
        params_sigma = torch.exp(params3)
        
        phi_prime = torch.exp(-0.5*torch.sum((x-params_mu)**2)/params_sigma**2)
        phi = phi_prime / 2
        
        q = torch.dot(params_gamma,phi)
        return q
        
        
    def forward(self,x):
        o1 = nn.Sequential(*self.layers_gamma)(x)
        o2 = nn.Sequential(*self.layers_mu)(x)
        o3 = nn.Sequential(*self.layers_sigma)(x)
        o = self.gaussian_mixture(x,o1,o2,o3)
        return o.squeeze(),params_gamma,params_mu,params_sigma
    
class q_phi_ff_connected(nn.Module):
    """
    Feed-forward neural network. 
    """
    
    def __init__(self,Nin,Nhid,Nout,num_layers=1):
        super(q_phi_ff_connected,self).__init__()
        
        # Auxiliary stuff
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=0)
        self.Nin = Nin
        self.Nc = int(Nout/(Nin+2))
        self.Nout = Nout
        self.actfun = nn.Sigmoid()
        self.pi = torch.tensor(np.pi)
                
        # Layers
        layers = nn.ModuleList()
        self.layer1 = nn.Linear(Nin,Nhid)
        self.layerlast = nn.Linear(Nhid,Nout)
        layers.append(self.layer1)
        layers.append(self.actfun)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(Nhid,Nhid))
            layers.append(self.actfun)
        layers.append(self.layerlast)
        self.layers = layers
    
    def gaussian_mixture(self,t,x,params):
        global params_gamma,params_mu,params_sigma,gammas,mus_max_indx
        """
        Computes a pdf that is a gaussian mixture given the weights, means and stdevs.
        The return is the logarithm of the pdf. 
        
        Parameters
        ----------
        t: tensor
            lablels to learn
        x : tensor
            input vector of the network.
        params : tensor
            means and stdevs of the product of gaussians.

        Returns
        -------
        tensor
            q_phi.
            
        """
        
        N = t.size()[1]
        gammas,mus,phis,M = [],[],[],params.size()[0]
        #print(f'M: {M}')
        for i in range(self.Nc):
            gc = params[:,i*(N+2):(i+1)*(N+2)]
            params_mu = gc[:,:N] # [M,Nc]
            params_gamma = gc[:,N:N+1] # [M,Nc*N]
            params_sigma = torch.exp(gc[:,N+1:N+2]) # [M,Nc]
            
            phi_prime = torch.exp(-0.5*torch.sum((t-params_mu)**2,dim=1)/params_sigma.squeeze(1)**2)
            phi = phi_prime / (((2*self.pi)**(N/2))*(params_sigma.squeeze(1)**N))
            gammas.append(params_gamma)
            mus.append(params_mu)
            phis.append(phi)
        
        gammas,mus,phis = self.softmax(torch.stack(gammas)),torch.stack(mus),torch.stack(phis)
        _,idcs = torch.max(gammas,0,keepdim=True)
        if self.Nc >1:
            mus_max_indx = torch.stack([mus[i][j] for i,j in zip(idcs.view(M),range(M))])
        else:
            mus_max_indx = mus
        
        gammas = gammas.view(M,1,self.Nc)
        phis = phis.view(M,self.Nc,1)
        q = torch.bmm(gammas,phis).squeeze(2)

        return q        
        
    def forward(self,t,x):
        o = nn.Sequential(*self.layers)(x)
        o = self.gaussian_mixture(t,x,o)
        return o.squeeze(),gammas.squeeze(2),params_mu,params_sigma,mus_max_indx
    
class q_phi_simple(nn.Module):
    """
    Feed-forward neural network. 
    """
    
    def __init__(self,Nin,Nhid,Nout,num_layers=1):
        super(q_phi_simple,self).__init__()
        
        # Auxiliary stuff
        self.actfun = nn.Sigmoid()
        self.softplus = nn.Softplus()
                
        # Layers
        layers = nn.ModuleList()
        self.layer1 = nn.Linear(Nin,Nhid)
        self.layerlast = nn.Linear(Nhid,Nout)
        layers.append(self.layer1)
        layers.append(self.actfun)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(Nhid,Nhid))
            layers.append(self.actfun)
        layers.append(self.layerlast)
        self.layers = layers
        
    def forward(self,x):
        o = nn.Sequential(*self.layers)(x)
        return self.softplus(o)

class q_phi_rnn(nn.Module):
    """
    LSTM MDN
    
    Parameters
    ----------
    input_size : int
        dimension of each element of the input sequence (1D in our case)
    hidden_size : int
        length of the output vector of the LSTM, h
    num_layers : int, optional
        number of stacked LSTM layers. Default is 1.
    Dense : bool, optional
        If true, adds a Linear layer that processes the output vector of 
        the LSTM and it returns the GMM parameters. Default is False.
    
    Returns
    -------
    q_phi : tensor
        (log of the) probability density of the input sequence x.
        
    
    """
    def __init__(self,input_size,hidden_size,num_layers=1,Dense=False):
        super(q_phi_rnn, self).__init__()
        
        # Auxiliary stuff
        self.pdf = self.gaussian_mixture if hidden_size==3*input_size else self.prod_of_gaussians
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax()
        self.Dense_bool = Dense
        
        # Layers    
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,
                            num_layers=num_layers,batch_first=True)
        self.Dense = nn.Linear(in_features=hidden_size,out_features=hidden_size)

    def prod_of_gaussians(self,x,params):
        """
        Computes a pdf that is a product of gaussians given the means and stdevs.
        The return is the logarithm of the pdf. 
        
        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        params : TYPE
            DESCRIPTION.

        Returns
        -------
        tensor
            Logarithm of q_phi.

        """
        global params_mu,params_sigma
        N = x.size()[1]
        n_params = params.size()[2]
        params_mu = params[:,:,:int(n_params/2)] # size = [n_samples,N,N]
        params_sigma = self.softplus(params[:,:,int(n_params/2):]) # size = [n_samples,N,N]
        sum_ = 0.5*(((x-params_mu)/params_sigma)**2)+torch.log(params_sigma) # size = [n_samples,N,N]
        logp = -(torch.sum(sum_,dim=1)+(N/2)*torch.log(torch.tensor(2*np.pi)).expand(x.size()[0],x.size()[2])) # size of return:
        return logp
    
    def gaussian_mixture(self,x,params):
        global params_gamma,params_mu,params_sigma
        N = x.size()[1]
        n_params = params.size()[2]
        params_gamma = self.softmax(params[:,:,:int(n_params/3)]) # size = [n_samples,N,N/3]
        params_mu = params[:,:,int(n_params/3):int(n_params*2/3)] # size = [n_samples,N,N/3]
        params_sigma = self.softplus(params[:,:,int(n_params*2/3):]) # size = [n_samples,N,N/3]
        exps = torch.exp(-0.5*(((x-params_mu)/params_sigma)**2))/(torch.sqrt(2*torch.tensor(np.pi))*params_sigma)
        p = torch.dot((params_gamma,exps),dim=1)
        logp = torch.log(p)
        return logp
        
    def forward(self,x):
        # x.size() = [n_samples,N,input_size]
        lstm_out, _ = self.lstm(x)
        lstm_out = self.Dense(lstm_out) if self.Dense_bool else lstm_out
        q_phi = self.pdf(x,lstm_out)
        return q_phi,params_mu,params_sigma
    
    
    
    
    
    
    
    
    
    