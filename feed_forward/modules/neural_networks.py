import torch
from torch import nn

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
        
        # We set the parameters 
        with torch.no_grad():
            self.lc1.weight = nn.Parameter(W1)
            self.lc1.bias = nn.Parameter(B)
            self.lc2.weight = nn.Parameter(W2)
   
    # We set the architecture
    def forward(self, x): 
        o = self.actfun(self.lc1(x))
        o = self.lc2(o)
        return o.squeeze()