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
class q_phi_ff(nn.Module):
    def __init__(self,Nin,Nhid,Nout,num_layers=1):
        super(q_phi_ff,self).__init__()
        
        self.actfun = nn.Sigmoid()
        
        layers_mu,layers_sigma = nn.ModuleList(),nn.ModuleList()
        layers_mu.append(nn.Linear(Nin,Nhid))
        layers_sigma.append(nn.Linear(Nin,Nhid))
        for _ in range(num_layers - 1):
            layers_mu.append(nn.Linear(Nhid,Nhid))
            layers_mu.append(self.actfun)
            layers_sigma.append(nn.Linear(Nhid,Nhid))
            layers_sigma.append(self.actfun)
        layers_mu.append(nn.Linear(Nhid,Nout))
        layers_sigma.append(nn.Linear(Nhid,Nout))
        self.layers_mu = layers_mu
        self.layers_sigma = layers_sigma
        self.softplus = nn.Softplus()
        self.N = Nin
        
        
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
        return -(torch.sum(sum_,dim=1)+self.N/2*torch.log(torch.tensor(2*math.pi)).repeat(x.size()[0]))
    
        
    def forward(self,x):
        o1 = nn.Sequential(*self.layers_mu)(x)
        o2 = nn.Sequential(*self.layers_sigma)(x)
        o = self.prod_of_gaussians(x,o1,o2)
        return o.squeeze(),params_mu,params_sigma
    

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
        self.softplus = nn.Softplus()
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
        # size of return: [n_samples,input_size]
        return -(torch.sum(sum_,dim=1)+(N/2)*torch.log(torch.tensor(2*np.pi)).expand(x.size()[0],x.size()[2]))
        
    def forward(self,x):
        # x.size() = [n_samples,N,input_size]
        lstm_out, _ = self.lstm(x)
        lstm_out = self.Dense(lstm_out) if self.Dense_bool else lstm_out
        q_phi = self.prod_of_gaussians(x,lstm_out)
        return q_phi,params_mu,params_sigma
    
    
    
    
    
    
    
    
    