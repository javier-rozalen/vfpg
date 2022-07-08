######################## IMPORTS ########################
import pathlib, os, sys
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('.')

import numpy as np
from tqdm import tqdm
import torch, math

# My modules
from modules.neural_networks import *

####################### PARAMETERS ################################
seed = 1
N = 5
input_size = 1
hidden_size = 2*input_size
n_samples = 3
num_layers = 1
Dense = False

####################### CODE ################################
torch.manual_seed(seed)
a = torch.randn(n_samples, N, input_size)
e1 = [[ 0.6614],
         [ 0.2669],
         [ 0.0617],
         [ 0.6213],
         [-0.4519]]
e2 = [[-0.1661],
 [-1.5228],
 [ 0.3817],
 [-1.0276],
 [-0.5631]]
e3 = [[-0.8923],
 [-0.0583],
 [-0.1955],
 [-0.9656],
 [ 0.4224]]
b = []
b.append(torch.tensor(e1))
b.append(torch.tensor(e2))
b.append(torch.tensor(e3))
b = torch.stack(b)

print(f'Input: {b}\n')
print(f'Size of the input: {b.size()}\n')

q_params_nn = q_phi_rnn(input_size=input_size, hidden_size=hidden_size,
                        num_layers=num_layers, Dense=Dense)
### Explanation of the inputs
# N --> length of the sequence (path)
# input_size --> number of features / dimensions of each element of the sequence (path)
# hidden_size --> dimensions of the output of the LSTM (as if it had just received a tensor of size 'input_size')
# num_layers --> number of layers for each gate
# Dense --> whether to add a Linear layer on top of each gate or not

o = q_params_nn(b)
q_phi = o[0]
params_mu = o[1]
params_sigma = o[2]
print(f'log(q_phi) = {q_phi}\n')
print(f'q_phi = {torch.exp(q_phi)}\n')
show_layers(q_params_nn)
print(count_params(q_params_nn))


########################## EXPLANATION OF THE CODE ##########################
"""
LSTMs must receive a 3D vector as input, the dimensions of which are: [n_samples,seq_length,dim_of_elements] or also
[seq_length,n_samples,dim_of_elements]. We will use the first option, which we specify by adding 'batch_first=True' 
in the LSTM construction (inside the class). We can easily create such vectors as done in variable 'a', but the useful
way for us is that of variable 'b'. Notice how a and b will be exactly the same. 

The next step is defining an instance of the LSTM; an explanation of the parameters needed to do so can be found in the
main code. In our case, passing the last two arguments is optional. 

Finally we call the instance on our input. There are prints everywhere that help understanding the output. 

Side note: we have tested that passing the n_samples to the LSTM sequentially (via for loop) and using the programmed
method yield identical results (as should be).

"""





















