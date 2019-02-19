#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:24:24 2019

@author: nsde
"""

#%%
import torch
from torch import nn
from torch import distributions as D 
from callbacks import callback_default, callback_moons, callback_mnist
from itertools import chain

#%%
class VAE_full_base(nn.Module):
    def __init__(self, lr):
        super(VAE_full_base, self).__init__()
        self.switch = 0.0
        self.lr = lr
        self.callback = callback_default()
        
    def init_optim(self):
        self.optimizer1 = torch.optim.Adam(chain(self.enc_mu.parameters(),
                                                 self.enc_std.parameters(),
                                                 self.dec_mu.parameters()), lr=self.lr)
        self.optimizer2 = torch.optim.Adam(self.dec_std.parameters(), lr=self.lr)
        
    def zero_grad(self):
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        
    def step(self):
        if self.switch==0:
            self.optimizer1.step()
        else:
            self.optimizer2.step()
    
    def encoder(self, x):
        return self.enc_mu(x), self.enc_std(x)
        
    def decoder(self, z):
        x_mu = self.dec_mu(z)
        x_std = self.switch*self.dec_std(z) + (1-self.switch)*torch.tensor(0.02**2)
        return x_mu, x_std
    
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
        p_dist = D.Independent(D.Normal(x_mu, x_std+epsilon), 1)
        
        # Calculate loss
        prior = D.Independent(D.Normal(torch.zeros_like(z),
                                       torch.ones_like(z)), 1)
        log_px = p_dist.log_prob(x)
        kl1 = q_dist.log_prob(z)
        kl2 = prior.log_prob(z)
        kl = kl1 - kl2
        elbo = (log_px - beta*kl)
        iw_elbo = elbo.logsumexp(dim=0) - torch.tensor(float(iw_samples)).log()
        
        return iw_elbo.mean(), log_px.mean(), kl.mean(), x_mu, x_std, z, z_mu, z_std

#%%
class VAE_full_moons(VAE_full_base):
    def __init__(self, lr):
        super(VAE_full_moons, self).__init__(lr=lr)
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
        self.dec_std = nn.Sequential(nn.Linear(2, 100), 
                                     nn.ReLU(), 
                                     nn.Linear(100, 2), 
                                     nn.Softplus())
        self.callback = callback_moons()
        self.latent_dim = 2
        
#%%
class VAE_full_mnist(VAE_full_base):
    def __init__(self, lr):
        super(VAE_full_mnist, self).__init__(lr=lr)
        self.enc_mu = nn.Sequential(nn.Linear(784, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 2))
        self.enc_std = nn.Sequential(nn.Linear(784, 256),
                                     nn.LeakyReLU(),
                                     nn.Linear(256, 128),
                                     nn.LeakyReLU(),
                                     nn.Linear(128, 2),
                                     nn.Softplus())
        self.dec_mu = nn.Sequential(nn.Linear(2, 128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 784),
                                    nn.ReLU())
        self.dec_std = nn.Sequential(nn.Linear(2, 128),
                                     nn.LeakyReLU(),
                                     nn.Linear(128, 256),
                                     nn.LeakyReLU(),
                                     nn.Linear(256, 784),
                                     nn.Softplus())
        self.callback = callback_mnist()
        self.latent_dim = 2