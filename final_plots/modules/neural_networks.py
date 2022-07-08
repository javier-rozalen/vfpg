
# -*- coding: utf-8 -*-

import torch, logging
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

toggle_prints = False

def print_(*message):
    if toggle_prints:
        print(*message)

class VAE(nn.Module):
    def __init__(self, dev, sample_size, batch_size, latent_size, MC_size, 
                 hidden_size_enc, hidden_size_dec):
        super(VAE, self).__init__()     
        
        # Names of dimensions (shapes):
            # latent_size = event_shape --> dimensions of latent variables z
            # sample_size --> dimensions of each example of the training set
            # hidden_size_enc/dec --> number of hidden neurons of enc/dec
            
        # Auxiliary stuff
        self.dev = dev
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.MC_size = MC_size
        
        # Encoder network
        self.enc_act = nn.Tanh()
        self.enc_shared = nn.Linear(in_features=sample_size,
                                    out_features=hidden_size_enc,
                                    bias=True)
        self.enc_mu = nn.Linear(in_features=hidden_size_enc,
                                out_features=latent_size,
                                bias=True)
        self.enc_log_var = nn.Linear(in_features=hidden_size_enc,
                                     out_features=latent_size,
                                     bias=True)
        
        # Decoder network
        self.dec_act = nn.Tanh()
        self.dec_shared = nn.Linear(in_features=latent_size,
                                out_features=hidden_size_dec,
                                bias=True)
        self.dec_mu = nn.Linear(in_features=hidden_size_dec,
                                out_features=sample_size,
                                bias=True)
        self.dec_log_var = nn.Linear(in_features=hidden_size_dec,
                                     out_features=sample_size,
                                     bias=True)
        
    def encoder(self, x):
        h_enc = self.enc_act(self.enc_shared(x))
        enc_mus = self.enc_mu(h_enc) # shape [batch_size, latent_size]
        enc_sigmas2 = torch.exp(self.enc_log_var(h_enc)) # shape idem
        
        return enc_mus, enc_sigmas2
    
    def encoder_sampler(self, enc_mus, enc_sigmas2): 
        cov_mat = torch.diag_embed(enc_sigmas2) 
        # cov_mat shape  = [batch_size, latent_size, latent_size]
        print_(f'covariance matrix: {cov_mat}', cov_mat.shape)
        encoder_dist = MultivariateNormal(loc=enc_mus, 
                                          covariance_matrix=cov_mat)
        z = encoder_dist.rsample(torch.Size([self.MC_size])).to(self.dev)
        # z shape = [MC_size, batch_size, latent_size]
        
        return z
    
    def decoder(self, z):
        h_dec = self.dec_act(self.dec_shared(z))
        dec_mus = self.dec_mu(h_dec) # shape [MC_size, batch_size, sample_size]
        dec_sigmas2 = torch.exp(self.dec_log_var(h_dec)) # shape idem
        print_(f'dec_mus: {dec_mus}', dec_mus.shape)
        print_(f'dec_sigmas2: {dec_sigmas2}', dec_sigmas2.shape)
        
        return dec_mus, dec_sigmas2 
    
    def decoder_sampler(self, z): 
        n_samples = z.size(0)
        h_dec = self.dec_act(self.dec_shared(z))
        dec_mus = self.dec_mu(h_dec) # shape [MC_size, batch_size, sample_size]
        dec_sigmas2 = torch.exp(self.dec_log_var(h_dec)) # shape idem
        cov_mat = torch.diag_embed(dec_sigmas2) 
        # cov_mat shape  = [batch_size, sample_size, sample_size]
        print_(f'covariance matrix: {cov_mat}', cov_mat.shape)
        decoder_dist = MultivariateNormal(loc=dec_mus, 
                                          covariance_matrix=cov_mat)
        x = decoder_dist.sample()  # paths
        # x shape = [batch_size, sample_size]
        
        return x
        
    def forward(self, x):
        # First we get a sample z (latent variable) from the encoder
        # x shape = [batch_size, sample_size]
        enc_mus, enc_sigmas2 = self.encoder(x)
        print_(f'x: {x}', x.shape)
        print_(f'enc_mus: {enc_mus}', enc_mus.shape)
        print_(f'enc_sigmas2: {enc_sigmas2}', enc_sigmas2.shape)
        z = self.encoder_sampler(enc_mus, enc_sigmas2)
        print_(f'sampled z: {z}', z.shape)
        
        dec_mus, dec_sigmas2 = self.decoder(z)
        
        return z, enc_mus, enc_sigmas2, dec_mus, dec_sigmas2
    
####################### TESTS #######################
if __name__ == '__main__':    
    print('Helloooo')