######################## IMPORTS ########################
import numpy as np
from tqdm import tqdm
import torch,math,os

# My modules
from modules.neural_networks import *
from modules.plotters import *
from modules.integration import simpson_weights
from modules.aux_functions import train_loop
from modules.physical_constants import *
from modules.dir_support import dir_support
from modules.loss_functions import *

######################## PARAMETERS ########################
# General parameters
N = 30
M = 10000
t_0 = 0.
t_f = 100.
eps = 5 # Maximum MC allowed error

# Names of files, directories
paths_file = f'../MonteCarlo/saved_data/paths_N{N}_M{M}.txt'
actions_file = f'../MonteCarlo/saved_data/actions_N{N}_M{M}.txt'
trained_models_path = 'saved_models/ff/'
trained_plots_path = 'saved_plots/ff/'

# Training parameters
n_epochs = 2000
Nhid = 50
Nin = N
Nc = 1
Nout = (N+2)*Nc
num_layers = 2
batch_size = 1000
learning_rate = 1e-3
epsilon = 1e-8
smoothing_constant = 0.9
seed = 1

# Plotting parameters
adaptive_factor = 2.5
leap = n_epochs/20

# Saves/Booleans
save_model = True
save_plot = False
show_periodic_plots = True
mini_batching = False
continue_from_last = False

model_name = f'N{N}_M{M}_nhid{Nhid}_Nc{Nc}_lr{learning_rate}_nlayers{num_layers}_seed{seed}'

######################## DATA FETCHING ########################
print('Fetching data...')
path_manifold,S_manifold = [],[]
with open(paths_file,'r') as file:
    for line in file.readlines():
        path = [torch.tensor(float(x)) for x in line.split(' ')]
        path_manifold.append(torch.stack(path))
    file.close()
with open(actions_file,'r') as file:
    for line in file.readlines():
        S = torch.tensor(float(line))
        S_manifold.append(S)
    file.close()
x_tensor = torch.stack(path_manifold)
S_tensor = torch.stack(S_manifold)
p_tensor = torch.exp(-S_tensor).unsqueeze(1)
p_tensor = S_tensor.unsqueeze(1)
M = len(path_manifold)
N = len(path_manifold[0])

print('Data fetching complete.\n')

x_test = x_tensor

######################## NN STUFF ########################
#torch.manual_seed(seed)

q_params_nn = q_phi_ff_connected(Nin,Nhid,Nout,num_layers=num_layers)
loss_fn = loss_neg_log_like
optimizer = torch.optim.RMSprop(params=q_params_nn.parameters(),lr=learning_rate,eps=epsilon)

if continue_from_last:
    q_params_nn.load_state_dict(torch.load(f'{trained_models_path}{model_name}.pt')['model_state_dict'])
    q_params_nn.eval()
    optimizer.load_state_dict(torch.load(f'{trained_models_path}{model_name}.pt')['optimizer_state_dict'])
    print('Resuming training from savepoint...\n')

######################## EPOCH LOOP ########################
x_axis,loss_list = [],[]
if not os.path.exists(trained_models_path+model_name+'.pt') or continue_from_last:
    if mini_batching:
        for t in tqdm(range(n_epochs)):
            for b in range(M//batch_size):
                x_tensor_batch = x_tensor[b*batch_size:(b+1)*batch_size]
                train_loop(x_tensor_batch,loss_fn,optimizer)
                loss_list.append(I.item())
                x_axis.append(t)
                if t == n_epochs-1 or (t+1)%leap==0:
                    if show_periodic_plots:
                        color = 'blue' if error<=eps/100 else 'red'
                        loss_plot(x_axis,loss_list,x_tensor,params_mu,params_sigma,I.item(),adaptive_factor,100*error,color,save=False)
                if save_plot and t == n_epochs-1:
                    dir_support([trained_plots_path])
                    loss_plot(x_axis,loss_list,x_tensor,params_mu,params_sigmaI.item(),adaptive_factor,100*error,color,save=True,
                              model_name=trained_plots_path+model_name+'.png')
    else:
        for t in tqdm(range(n_epochs)):
            things = train_loop(model=q_params_nn,train_set=x_tensor,target_data=p_tensor,
                           loss_fn=loss_fn,optimizer=optimizer)
            I = things[0]
            loss_list.append(I.item())
            x_axis.append(t)
            if t == n_epochs-1 or (t+1)%leap==0:
                if show_periodic_plots:
                    pred_q,pred_gamma,pred_mu,pred_sigma,mu_max_idx = q_params_nn(p_tensor,x_test)
                    loss_plot(x_axis,loss_list,I.item(),adaptive_factor)
            if save_plot and t == n_epochs-1:
                dir_support([trained_plots_path])
                pred_q,pred_gamma,pred_mu,pred_sigma = q_params_nn(p_tensor,x_test)
                loss_plot_testos(x_axis,loss_list,I.item(),adaptive_factor,x_test,mu_max_idx.detach())

    print('\nDone! :)')
    if save_plot:
        print(f'Plot saved in {trained_plots_path}')
        
    if save_model:
        dir_support([trained_models_path])
        state_dict = {'model_state_dict':q_params_nn.state_dict(),
                     'optimizer_state_dict':optimizer.state_dict(),
                     'epochs':n_epochs,
                     'Nin':Nin,
                     'Nhid':Nhid,
                     'Nout':Nout,
                     'Nc':Nc,
                     'num_layers':num_layers}
        torch.save(state_dict,trained_models_path+model_name+'.pt')
        print(f'Model saved in {trained_models_path}')
    
else:
    print(f'Model {model_name} has already been trained. Skipping...')
    
    
    
    