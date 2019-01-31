#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:24:43 2019

@author: nsde
"""

#%%
import torch
from torch import nn
from torch import distributions as D

#%%
class singlestd(nn.Module):
    def __init__(self):
        super(singlestd, self).__init__()
        self.std = nn.Parameter(torch.tensor(0.02, requires_grad=True))
    
    def forward(self, z):
        return self.std

#%%
class VAE_single(nn.Module):
    def __init__(self, ):
        super(VAE_single, self).__init__()
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
        self.dec_std = nn.Sequential(singlestd(),
                                     nn.Softplus())
        
    def forward(self, x, beta=1.0, switch=1.0):
        # Encoder step
        z_mu, z_std = self.enc_mu(x), self.enc_std(x)
        q_dist = D.Independent(D.Normal(z_mu, z_std), 1)
        z = q_dist.rsample()
        
        # Decoder step
        x_mu, x_std = self.dec_mu(z), self.dec_std(z)
        x_std = switch*x_std + (1-switch)*torch.tensor(0.02)
        p_dist = D.Independent(D.Normal(x_mu, x_std), 1)
        
        # Calculate loss
        prior = D.Independent(D.Normal(torch.zeros_like(z),
                                       torch.ones_like(z)), 1)
        log_px = p_dist.log_prob(x)
        kl = q_dist.log_prob(z) - prior.log_prob(z)
        elbo = (log_px - beta*kl).mean()
        
        return elbo, x_mu, x_std, z, z_mu, z_std