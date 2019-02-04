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
        self.adverserial = nn.Sequential(nn.Linear(2, 1000),
                                         nn.ReLU(),
                                         nn.Linear(1000, 1000),
                                         nn.ReLU(),
                                         nn.Linear(1000, 1),
                                         nn.Sigmoid())
        
    def encoder(self, x):
        return self.enc_mu(x), self.enc_std(x)
        
    def decoder(self, z, switch=1.0):
        x_mu = self.dec_mu(z)
        
        prop = self.adverserial(x_mu)
        x_std = 1.0 / (prop+1e-6)
        
        return x_mu, switch*x_std+(1-switch)*torch.tensor(0.02**2)
    
    def forward(self, x, beta=1.0, switch=1.0, iw_samples=1):
        # Encoder step
        z_mu, z_std = self.encoder(x)
        q_dist = D.Independent(D.Normal(z_mu, z_std), 1)
        z = q_dist.rsample([iw_samples])
        
        # Decoder step
        x_mu, x_std = self.decoder(z, switch)
        
        if switch:
            valid = torch.ones((x.shape[0], 1),  device = x.device)
            fake = torch.zeros((x.shape[0], 1), device = x.device)
            labels = torch.cat([valid, fake], dim=0)
            x_cat = torch.cat([x, x_mu[0]], dim=0)
            
            prop = self.adverserial(x_cat)
            advert_loss = F.binary_cross_entropy(prop, labels, reduction='sum')
            x_std = 1.0 / (prop+1e-6)
        else:
            advert_loss = 0
        
        p_dist = D.Independent(D.Normal(x_mu[0], x_std[:x.shape[0]]), 1)
        
        # Calculate loss
        prior = D.Independent(D.Normal(torch.zeros_like(z),
                                       torch.ones_like(z)), 1)
        log_px = p_dist.log_prob(x)
        kl = q_dist.log_prob(z) - prior.log_prob(z)
        elbo = (log_px - beta*kl).mean()
        iw_elbo = elbo.logsumexp(dim=0) - torch.tensor(float(iw_samples)).log()
        
        return iw_elbo.mean() - 10*advert_loss, log_px.mean(), kl.mean(), x_mu[0], x_std, z[0], z_mu, z_std
    