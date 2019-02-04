#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:25:16 2019

@author: nsde
"""

#%%
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D

#%%
def dist(X, Y): # X: ? x N x d , Y: ? x M x d
    dist =  X.norm(p=2, dim=-1, keepdim=True)**2 + \
            Y.norm(p=2, dim=-1, keepdim=False)**2 - \
            2*torch.matmul(X, Y.t())
    return dist # ? x N x M

#%%
class VAE_rbf(nn.Module):
    def __init__(self, ):
        super(VAE_rbf, self).__init__()
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
        self.C = nn.Parameter(torch.randn(30, 2))
        self.W = nn.Parameter(torch.rand(30,2))
        #self.lamb = 5
        self.alpha = nn.Parameter(torch.rand(30,)+10)
        self.lamb = nn.Parameter(5*torch.rand(30,)+5)
    
    def encoder(self, x):
        return self.enc_mu(x), self.enc_std(x)
    
    def decoder(self, z, switch=1.0):
        x_mu = self.dec_mu(z)
        kernel = (self.alpha**2)*torch.exp(-F.softplus(self.lamb) * dist(z, self.C))
        inv_std = torch.matmul(kernel, F.softplus(self.W)) + 1e-10
        x_std = switch * (1.0/inv_std) + (1-switch)*torch.tensor(0.02**2)
        return x_mu, x_std
    
    def forward(self, x, beta=1.0, switch=1.0, iw_samples=1):
        # Encoder step
        z_mu, z_std = self.encoder(x)
        q_dist = D.Independent(D.Normal(z_mu, z_std), 1)
        z = q_dist.rsample([iw_samples])
        
        # Decoder step
        x_mu, x_std = self.decoder(z, switch)
        p_dist = D.Independent(D.Normal(x_mu, x_std), 1) 
        
        # Calculate loss
        prior = D.Independent(D.Normal(torch.zeros_like(z),
                                       torch.ones_like(z)), 1)
        log_px = p_dist.log_prob(x)
        kl = q_dist.log_prob(z) - prior.log_prob(z)
        elbo = (log_px - beta*kl).mean()
        iw_elbo = elbo.logsumexp(dim=0) - torch.tensor(float(iw_samples)).log()
        
        return iw_elbo.mean(), log_px.mean(), kl.mean(), x_mu[0], x_std[0], z[0], z_mu, z_std