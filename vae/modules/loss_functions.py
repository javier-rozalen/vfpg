######### IMPORTS ##########
import torch, time
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from modules.physical_constants import m, w, hbar
from torch.distributions.normal import Normal
###########################

toggle_prints = False

def print_(*message):
    if toggle_prints:
        print(*message)
        
print_('\n')
############################ AUXILIARY FUNCTIONS ##############################
def ELBO(vae, train_data):
    
    # We get the encoder and decoder variational parameters and also the latent
    # space variable z sampled from the encoder
    z, enc_mus, enc_sigmas, dec_mus, dec_sigmas = vae(train_data)
    
    # Prior of z is a Normal(0, 1)
    prior_z_dist = Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z))
    
    # Encoder prob is a Normal(enc_mus, enc_sigmas)
    enc_dist = Normal(loc=enc_mus, scale=enc_sigmas)
    
    # Decoder prob is a Normal(dec_mus, dec_sigmas)
    dec_dist = Normal(loc=dec_mus, scale=dec_sigmas)
    
    # Reconstruction Loss
    log_p = dec_dist.log_prob(train_data)
    
    # Regularization term (KL-divergence)
    log_p_z = prior_z_dist.log_prob(z)
    log_q = enc_dist.log_prob(z)
    kl = log_p_z - log_q
    
    # ELBO Loss
    elbo_loss = torch.sum(kl + log_p, dim=1)
    MC_error = 0.
    
    return elbo_loss, MC_error