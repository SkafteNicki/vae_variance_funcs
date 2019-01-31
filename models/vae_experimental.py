# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:59:10 2019

@author: nsde
"""

#%%
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D

#%%
class VAE_experimental(nn.Module):
    def __init__(self, ):
        super(VAE_experimental, self).__init__()
        self.enc_mu = nn.Sequential(nn.Linear(2, 100), 
                                    nn.ReLU(), 
                                    nn.Linear(100, 2))
        self.enc_std = nn.Sequential(nn.Linear(2, 100), 
                                     nn.ReLU(), 
                                     nn.Linear(100, 2), 
                                     nn.Softplus())
        self.dec_mu = nn.Sequential(nn.Linear(2, 100), 
                                    nn.ReLU(), 
                                    nn.Linear(100, 2))
        
    def forward(self, x, beta=1.0, switch=1.0):
        # Encoder step 1
        z_mu, z_std = self.enc_mu(x), self.enc_std(x)
        q_dist = D.Independent(D.Normal(z_mu, z_std), 1)
        z = q_dist.rsample()
        
        # Decoder step 1
        x_mu = self.dec_mu(z)
        
        # Encoder step 2
        z_mu2 = self.enc_mu(x_mu)
        x_std = (z_mu - z_mu2).norm(dim=1, keepdim=True)
        x_std = switch * x_std + (1-switch)*torch.tensor(0.02)
        
        p_dist = D.Independent(D.Normal(x_mu, x_std), 1) 
        
        # Calculate loss
        prior = D.Independent(D.Normal(torch.zeros_like(z),
                                       torch.ones_like(z)), 1)
        log_px = p_dist.log_prob(x)
        kl = q_dist.log_prob(z) - prior.log_prob(z)
        elbo = (log_px - beta*kl).mean()
        
        return elbo, x_mu, x_std, z, z_mu, z_std