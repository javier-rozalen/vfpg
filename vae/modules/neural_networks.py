
# -*- coding: utf-8 -*-

import torch, logging
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np

toggle_prints = False

def print_(*message):
    if toggle_prints:
        print(*message)

class VAE(nn.Module):
    def __init__(self, sample_size, latent_size, hidden_size_enc, 
                 hidden_size_dec):
        super(VAE, self).__init__()
        # Auxiliary stuff

        
        
        # Encoder network
        self.enc_act = nn.Tanh()
        self.enc_l1 = nn.Linear(in_features=sample_size,
                                out_features=hidden_size_enc,
                                bias=True)
        self.enc_l2_mu = nn.Linear(in_features=hidden_size_enc,
                                   out_features=latent_size,
                                   bias=True)
        self.enc_l2_sigma2 = nn.Linear(in_features=hidden_size_enc,
                                       out_features=latent_size,
                                       bias=True)
        
        # Decoder network
        self.dec_act = nn.Tanh()
        self.dec_l1 = nn.Linear(in_features=latent_size,
                                out_features=hidden_size_dec,
                                bias=True)
        self.dec_l2_mu = nn.Linear(in_features=hidden_size_dec,
                                   out_features=sample_size,
                                   bias=True)
        self.dec_l2_sigma2 = nn.Linear(in_features=hidden_size_dec,
                                       out_features=latent_size,
                                       bias=True)
        
    def encoder(self, x):
        h_enc = self.enc_act(self.enc_l1(x))
        enc_mus = self.enc_l2_mu(h_enc)
        enc_sigmas = self.enc_l2_sigma2(h_enc)
        
        return enc_mus, enc_sigmas
    
    def encoder_sampler(self, enc_mus, enc_sigmas):
        encoder_dist = Normal(loc=enc_mus, scale=enc_sigmas)
        z = encoder_dist.rsample()
        
        return z
    
    def decoder(self, z):
        h_dec = self.dec_act(self.dec_l1(z))
        dec_mus = self.dec_l2_mu(h_dec)
        dec_sigmas = self.dec_l2_sigma2(h_dec)
        
        return dec_mus, dec_sigmas     
        
    def forward(self, x, z):
        # First we get a sample z (latent variable) from the encoder
        enc_mus, enc_sigmas = self.encoder(x)
        z = self.encoder_sampler(enc_mus, enc_sigmas)
        
        dec_mus, dec_sigmas = self.decoder(z)
        
        return z, enc_mus, enc_sigmas, dec_mus, dec_sigmas
    
####################### TESTS #######################
if __name__ == '__main__':    
    print('Helloooo')