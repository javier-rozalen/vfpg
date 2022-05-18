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
         
        # We set the operators 
        self.lc1 = nn.Linear(Nin,Nhid,bias=True) # shape = (Nhid,Nin)
        self.actfun = nn.Sigmoid() # activation function
        self.lc2 = nn.Linear(Nhid,Nout,bias=False) # shape = (Nout,Nhid)
        self.softplus = nn.Softplus()
        
        # We set the parameters 
        with torch.no_grad():
            self.lc1.weight = nn.Parameter(W1)
            self.lc1.bias = nn.Parameter(B)
            self.lc2.weight = nn.Parameter(W2)
            
    def gaussian_mixture(self,x,params):
        return torch.sum(params)
    
    def prod_of_gaussians(self,x,params):
        params_mu = params[:int(len(params)/2)]
        params_sigma = self.softplus(params[int(len(params)/2):])
        gaussians = (1/(torch.sqrt(2*torch.tensor(math.pi))*params_mu))*torch.exp(-0.5*((x-params_mu)/params_sigma)**2)
        print(gaussians)
        return torch.prod(gaussians)
   
    # We set the architecture
    def forward(self, x): 
        o = self.actfun(self.lc1(x))
        o = self.lc2(o)
        o = self.prod_of_gaussians(x,o)
        return o.squeeze()