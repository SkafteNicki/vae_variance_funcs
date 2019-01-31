# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 15:35:32 2019

@author: nsde
"""

#%%
import torch
from torch import nn
from torch import distributions as D

#%%
def dist(X, Y): # X: N x d , Y: M x d
    dist =  X.norm(p=2, dim=1, keepdim=True)**2 + \
            Y.norm(p=2, dim=1, keepdim=False)**2 - \
            2*torch.mm(X, Y.t())
    return dist # N x M

#%%
class VAE_student_rbf(nn.Module):
    def __init__(self, ):
        super(VAE_student_rbf, self).__init__()
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
        self.dec_df = nn.Sequential(nn.Linear(2, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 2),
                                    nn.Softplus())
        self.C = nn.Parameter(torch.randn(30, 2))
        self.W = nn.Parameter(torch.rand(30,2))
        self.lamb = 5
        
    def forward(self, x, beta=1.0, switch=1.0):
        # Encoder step
        z_mu, z_std = self.enc_mu(x), self.enc_std(x)
        q_dist = D.Independent(D.Normal(z_mu, z_std.sqrt()), 1)
        z = q_dist.rsample()
        
        # Decoder step
        x_mu, x_df = self.dec_mu(z), self.dec_df(z)
        inv_std = torch.mm(torch.exp(-self.lamb * dist(z, self.C)), torch.clamp(self.W, min=0.0)) + 1e-10
        x_scale = switch * (1/inv_std) + (1-switch)*torch.tensor(0.02)
        p_dist = D.Independent(D.StudentT(x_df, x_mu, x_scale), 1)
        
        # Calculate loss
        prior = D.Independent(D.Normal(torch.zeros_like(z),
                                       torch.ones_like(z)), 1)
        log_px = p_dist.log_prob(x)
        kl = q_dist.log_prob(z) - prior.log_prob(z)
        elbo = (log_px - beta*kl).mean()
        
        return elbo, x_mu, x_scale, z, z_mu, z_std