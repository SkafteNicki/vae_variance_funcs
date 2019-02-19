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
from callbacks import callback_default, callback_moons

#%%
class VAE_student_base(nn.Module):
    def __init__(self, lr):
        super(VAE_student_base, self).__init__()
        self.switch = 0.0
        self.lr = lr
        self.callback = callback_default()
    
    def init_optim(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def step(self):
        self.optimizer.step()    
    
    def encoder(self, x):
        return self.enc_mu(x), self.enc_std(x)        
    
    def decoder(self, z):
        return self.dec_mu(z), self.dec_std(z)
    
    def sample(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=device)
            x_mu, _ = self.decoder(z)
            return x_mu
    
    def forward(self, x, beta=1.0, iw_samples=1, epsilon=1e-5):
        # Encoder step
        z_mu, z_std = self.encoder(x)
        q_dist = D.Independent(D.Normal(z_mu, z_std+epsilon), 1)
        z = q_dist.rsample([iw_samples])
        
        # Decoder step
        x_mu, x_std = self.decoder(z)
        x_df = self.dec_df(z)
        p_dist = D.Independent(D.StudentT(x_df, x_mu, x_std+epsilon), 1)
        
        # Calculate loss
        prior = D.Independent(D.Normal(torch.zeros_like(z),
                                       torch.ones_like(z)), 1)
        log_px = p_dist.log_prob(x)
        kl = q_dist.log_prob(z) - prior.log_prob(z)
        elbo = (log_px - beta*kl).mean()
        iw_elbo = elbo.logsumexp(dim=0) - torch.tensor(float(iw_samples)).log()
        
        return iw_elbo.mean(), log_px.mean(), kl.mean(), x_mu, x_std, z, z_mu, z_std
    
#%%
class VAE_student_moons(VAE_student_base):
    def __init__(self, lr):
        super(VAE_student_moons, self).__init__(lr=lr)
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
        self.dec_std = nn.Sequential(nn.Linear(2, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 2),
                                    nn.Softplus())
        self.callback = callback_moons()
        self.latent_dim = 2