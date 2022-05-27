######### IMPORTS ##########
import torch,math
from torch import nn
###########################

############################ AUXILIARY FUNCTIONS ##############################
def show_layers(model):
    print("\nLayers and parameters:\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:100]} \n")
        
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

############################## NEURAL NETWORKS ################################
class q_phi_layers(nn.Module):
    def __init__(self,Nin,Nhid,Nout,num_layers=1):
        super(q_phi_layers,self).__init__()
        
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
        """
        Computes a pdf that is a product of gaussians given the means and stdevs.
        
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
        return o.squeeze()
    
    
    
    
    
    
    
    
    