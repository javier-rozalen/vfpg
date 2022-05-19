######### IMPORTS ##########
import torch,math
from torch import nn
###########################

def show_layers(model):
    print("\nLayers and parameters:\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:100]} \n")
        
class q_phi(nn.Module):   
    def __init__(self,Nin,Nhid,Nout,W1,W2,B):       
        super(q_phi, self).__init__()
         
        self.actfun = nn.Sigmoid() # activation function
        # NN_mu
        self.lc1 = nn.Linear(Nin,Nhid,bias=True) # shape = (Nhid,Nin)
        self.lc2 = nn.Linear(Nhid,Nout,bias=False) # shape = (Nout,Nhid)
        
        # NN_sigma
        self.lc3 = nn.Linear(Nin,Nhid,bias=True) # shape = (Nhid,Nin)
        self.lc4 = nn.Linear(Nhid,Nout,bias=False) # shape = (Nout,Nhid)
        
        # Others
        self.softplus = nn.Softplus()
        self.N = Nin
        
        
        # We set the parameters 
        with torch.no_grad():
            """
            self.lc1.weight = nn.Parameter(W1)
            self.lc1.bias = nn.Parameter(B)
            self.lc2.weight = nn.Parameter(W2)
            
            self.lc3.weight = nn.Parameter(W1)
            self.lc3.bias = nn.Parameter(B)
            self.lc4.weight = nn.Parameter(W2)"""
            
    def gaussian_mixture(self,x,params):
        
        return torch.sum(params)
    
    def prod_of_gaussians(self,x,params1,params2):
        """
        
        
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
        sum_ = 0.5*(((x-params_mu)/params_sigma)**2)-torch.log(params_sigma)
        return torch.sum(sum_,dim=1)-self.N/2*torch.log(torch.tensor(2*math.pi)).repeat(x.size()[0])
   
    # We set the architecture
    def forward(self, x): 
        o1 = self.lc2(self.actfun(self.lc1(x))) 
        o2 = self.lc4(self.actfun(self.lc3(x)))
        o = self.prod_of_gaussians(x,o1,o2)
        return o.squeeze()