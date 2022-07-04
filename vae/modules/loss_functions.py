######### IMPORTS ##########
import torch, time
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from modules.physical_constants import m, w, hbar
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
###########################

toggle_prints = False

def print_(*message):
    if toggle_prints:
        print(*message)
        
############################ LOSS FUNCTIONS ##############################
def ELBO(vae, train_data, dev):
    
    # We get the encoder and decoder variational parameters and also the latent
    # space variable z sampled from the encoder
    z, enc_mus, enc_sigmas2, dec_mus, dec_sigmas2 = vae(train_data)
    
    MC_size, batch_size, latent_size = z.size(0), z.size(1), z.size(2)

    # Prior of z is a Normal(0, 1)
    prior_z_loc = torch.zeros(batch_size, latent_size).to(dev)
    prior_z_cov_mat = torch.eye(latent_size).reshape(1, latent_size, latent_size)
    prior_z_cov_mat = prior_z_cov_mat.repeat(batch_size, 1, 1).to(dev)
    print_(f'prior_z loc: {prior_z_loc}', prior_z_loc.shape)
    print_(f'prior_z_cov_mat: {prior_z_cov_mat}', prior_z_cov_mat.shape)
    prior_z_dist = MultivariateNormal(loc=prior_z_loc, 
                                      covariance_matrix=prior_z_cov_mat)
    
    # Encoder prob is a Normal(enc_mus, enc_sigmas)
    enc_cov_mat = torch.diag_embed(enc_sigmas2)
    enc_dist = MultivariateNormal(loc=enc_mus, 
                                  covariance_matrix=enc_cov_mat)
    print_(f'enc_loc: {enc_mus}', enc_mus.shape)
    print_(f'enc_cov_mat: {enc_cov_mat}', enc_cov_mat.shape)
    
    # Decoder prob is a Normal(dec_mus, dec_sigmas)
    dec_cov_mat = torch.diag_embed(dec_sigmas2)
    dec_dist = MultivariateNormal(loc=dec_mus, 
                                  covariance_matrix=dec_cov_mat)
    print_(f'dec_loc: {dec_mus}', dec_mus.shape)
    print_(f'dec_cov_mat: {dec_cov_mat}', dec_cov_mat.shape)
    
    # Reconstruction Loss
    log_p = dec_dist.log_prob(train_data) # shape [MC_size, batch_size]
    print_(f'log_p: {log_p}', log_p.shape)
    
    # Regularization term (KL-divergence)
    log_p_z = prior_z_dist.log_prob(z) # shape [MC_size, batch_size]
    log_q = enc_dist.log_prob(z) # shape [MC_size, batch_size]
    kl = log_p_z - log_q
    print_(f'log_p_z: {log_p_z}', log_p_z.shape)
    print_(f'log_q: {log_q}', log_q.shape)
    print_(f'kl: {kl}', kl.shape)
    
    # ELBO Loss
    elbo_loss = torch.sum(torch.sum(kl + log_p, dim=0)) / (batch_size * MC_size)
    
    # MonteCarlo error
    M = z.size(0)
    MC_error = torch.sqrt((torch.sum(torch.sum((kl+log_p)**2, dim=0))/M-elbo_loss**2) / M)
    
    return -elbo_loss, MC_error