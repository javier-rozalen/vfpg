
# -*- coding: utf-8 -*-
####################### IMPORTS #######################
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np

####################### NEURAL NETWORKS #######################
class VFPG_ours(nn.Module):

    def __init__(self, dev, M, N, input_size, nhid, hidden_size, out_size, 
                 num_layers, Dense=True):
        super(VFPG_ours, self).__init__()
        
        # Auxiliary stuff
        self.dev = dev
        self.M = M
        self.N = N # length of each path
        self.input_size = input_size
        self.nhid = nhid # number of hidden neurons 
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.Nc = int(out_size / (1 + 2))
        self.Dense = Dense
        self.pi = torch.tensor(np.pi)
        self.softmax = nn.Softmax(dim=1)
        
        # Neural Network
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc = nn.Linear(in_features=hidden_size, 
                            out_features=out_size,                                                   
                            bias=True)

    def GMM(self, params):    
        # params size = [M, (N+2)*Nc]
        
        gammas = self.softmax(params[:, :self.Nc]) # size [M, Nc]
        sigmas = torch.exp(params[:, self.Nc:2*self.Nc]) # size [M, Nc]
        mus = params[:, 2*self.Nc:] # size [M, N*Nc]
        """
        print(f'gammas: {gammas}', gammas.shape)
        print(f'sigmas: {sigmas}', sigmas.shape)
        print(f'mus: {mus}', mus.shape)
        """
        return gammas, mus, sigmas 
    
    def GMM_sampler(self, gammas, mus, sigmas):
        
        # Sampling from the mixture density
        g_sampled_idcs = Categorical(gammas).sample().unsqueeze(1)
        chosen_mus = mus.gather(1, g_sampled_idcs)
        chosen_sigmas = sigmas.gather(1, g_sampled_idcs)
        gmm_dist = Normal(loc=chosen_mus, scale=chosen_sigmas)
        x_pred = gmm_dist.sample()
    
        # Computing the probability of the sample
        kernels_exponent = (x_pred.repeat(1, self.Nc) - mus)**2
        kernels = torch.exp(-0.5*kernels_exponent) / (sigmas**2)
        kernels /= ((2*self.pi)**(self.N/2))*(sigmas**self.N)
        
        gammas = gammas.view(self.M, 1, self.Nc)
        kernels = kernels.view(self.M, self.Nc, 1)
        x_cond_prob = (gammas @ kernels).squeeze(2)

        return x_pred.unsqueeze(2), x_cond_prob 
        
    def forward(self, x_prev):
        preds, cond_probs = [x_prev], [torch.ones(self.M, 1).to(self.dev)]

        # Initial cell states
        h_t = torch.zeros(self.num_layers, self.M, self.hidden_size).to(self.dev)
        c_t = torch.zeros(self.num_layers, self.M, self.hidden_size).to(self.dev)
        
        # Recurrence loop
        for i in range(self.N - 2):
            h_last_layer, (h_t, c_t) = self.lstm(x_prev, (h_t, c_t))
            y = self.fc(h_last_layer) if self.Dense else h_last_layer
            gammas, mus, sigmas = self.GMM(y.squeeze(1))

            x_pred, x_cond_prob = self.GMM_sampler(gammas, mus, sigmas)
            #print(f'x_pred: {x_pred}', x_pred.shape)
            preds.append(x_pred)
            cond_probs.append(x_cond_prob)
            x_prev = x_pred
            x_prev = torch.randn(self.M, 1, self.input_size) 
        
        # We close the paths
        preds.append(preds[0])
        cond_probs.append(cond_probs[0])
        paths = torch.stack(preds).squeeze().transpose(0,1)
        paths_cond_probs = torch.stack(cond_probs).squeeze().transpose(0,1)
        #print(f'paths: {paths}', paths.shape)
        #print(f'paths_cond_probs: {paths_cond_probs}', paths_cond_probs.shape)

        return paths, paths_cond_probs
    
class VFPG_theirs(nn.Module):

    def __init__(self, dev, M, N, input_size, nhid, hidden_size, out_size, 
                 num_layers, Dense=True):
        super(VFPG_theirs, self).__init__()
        
        # Auxiliary stuff
        self.dev = dev
        self.M = M
        self.N = N # length of each path
        self.input_size = input_size
        self.nhid = nhid # number of hidden neurons 
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.Nc = int(out_size / (1 + 2))
        self.Dense = Dense
        self.pi = torch.tensor(np.pi)
        self.softmax = nn.Softmax(dim=2)
        
        # Neural Network
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, 
                            out_features=out_size,                                                   
                            bias=True)

    def GMM(self, params):    
        # params size = [M, N, (N+2)*Nc]
        gammas = self.softmax(params[:, :, :self.Nc]) # size [M, N, Nc]
        sigmas = torch.exp(params[:, :, self.Nc:2*self.Nc]) # size [M, N, Nc]
        mus = params[:, :, 2*self.Nc:] # size [M, N, N*Nc]
        """
        print(f'gammas: {gammas}', gammas.shape)
        print(f'sigmas: {sigmas}', sigmas.shape)
        print(f'mus: {mus}', mus.shape)
        """
        return gammas, mus, sigmas 
    
    def GMM_sampler(self, gammas, mus, sigmas):
        
        # Sampling from the mixture density
        g_sampled_idcs = Categorical(gammas).sample().unsqueeze(2)
        #print(f'g_sampled_idcs: {g_sampled_idcs}', g_sampled_idcs.shape)
        chosen_mus = mus.gather(2, g_sampled_idcs)
        chosen_sigmas = sigmas.gather(2, g_sampled_idcs)
        #print(f'chosen_mus: {chosen_mus}', chosen_mus.shape)
        gmm_dist = Normal(loc=chosen_mus, scale=chosen_sigmas)
        x_pred = gmm_dist.sample()
        #print(f'x_pred: {x_pred}', x_pred.shape)
    
        # Computing the probability of the sample
        #print(f'x_pred_rep: {x_pred.repeat(1, 1, self.Nc)}', x_pred.repeat(1, 1, self.Nc).shape)
        kernels_exponent = (x_pred.repeat(1, 1, self.Nc) - mus)**2
        kernels = torch.exp(-0.5*kernels_exponent) / (sigmas**2)
        kernels /= ((2*self.pi)**(self.N/2))*(sigmas**self.N)
        
        _gammas = gammas.view(self.M, self.N, 1, self.Nc)
        _kernels = kernels.view(self.M, self.N, self.Nc, 1)
        #print(f'_gammas: {_gammas}', _gammas.shape)
        #print(f'_kernels: {_kernels}', _kernels.shape)
        x_cond_prob = (_gammas @ _kernels).squeeze(3).squeeze(2)
        #print(f'x_cond_prob: {x_cond_prob}', x_cond_prob.shape)

        return x_pred, x_cond_prob
        
    def forward(self, z):
        
        # Initial cell states
        h_t = torch.zeros(self.num_layers, self.M, self.hidden_size).to(self.dev)
        c_t = torch.zeros(self.num_layers, self.M, self.hidden_size).to(self.dev)
        
        h_last_layer, (h_t, c_t) = self.lstm(z, (h_t, c_t)) # LSTM
        """
        print(f'h_last_layer: {h_last_layer}', h_last_layer.shape)
        print(f'h_t: {h_t}', h_t.shape)
        print(f'c_t: {c_t}', c_t.shape)
        """
        y = self.fc(h_last_layer) if self.Dense else h_last_layer # Linear
        #print(f'y: {y}', y.shape)
        gammas, mus, sigmas = self.GMM(y) # mixture params computation
        paths, paths_cond_probs = self.GMM_sampler(gammas, mus, sigmas) # samples
        """
        print(f'gammas: {gammas}', gammas.shape)
        print(f'first gammas: {gammas[:, 0, :]}', gammas[:, 0, :].shape)
        """
        return paths, paths_cond_probs
    
####################### TESTS #######################
if __name__ == '__main__':    
    print('Helloooo')