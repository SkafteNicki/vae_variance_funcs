#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:24:54 2019

@author: nsde
"""

#%%
import torch
from torch import nn
from torch import distributions as D

#%%
class VAE_student(nn.Module):
    def __init__(self, ):
        super(VAE_student, self).__init__()
        self.enc_mu = nn.Sequential(nn.Linear(2, 100), 
                                    nn.ReLU(), 
                                    nn.Linear(100, 2))
        self.enc_var = nn.Sequential(nn.Linear(2, 100), 
                                     nn.ReLU(), 
                                     nn.Linear(100, 2), 
                                     nn.Softplus())
        self.dec_mu = nn.Sequential(nn.Linear(2, 100), 
                                    nn.ReLU(), 
                                    nn.Linear(100, 2))
        self.dec_df = nn.Sequential(nn.Linear(2, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 2),
                                    nn.Softplus())
        self.dec_scale = nn.Sequential(nn.Linear(2, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 2),
                                    nn.Softplus())
        
    def forward(self, x, beta=1.0, switch=1.0):
        # Encoder step
        z_mu, z_var = self.enc_mu(x), self.enc_var(x)
        q_dist = D.Independent(D.Normal(z_mu, z_var.sqrt()), 1)
        z = q_dist.rsample()
        
        # Decoder step
        x_mu, x_df, x_scale = self.dec_mu(z), self.dec_df(z), self.dec_scale(z)
        p_dist = D.Independent(D.StudentT(x_df, x_mu, x_scale), 1)
        
        # Calculate loss
        prior = D.Independent(D.Normal(torch.zeros_like(z),
                                       torch.ones_like(z)), 1)
        log_px = p_dist.log_prob(x)
        kl = q_dist.log_prob(z) - prior.log_prob(z)
        elbo = (log_px - beta*kl).mean()
        
        return elbo, x_mu, x_scale, z, z_mu, z_var