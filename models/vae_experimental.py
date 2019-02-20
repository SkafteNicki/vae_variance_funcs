# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:59:10 2019

@author: nsde
"""

#%%
import torch
from torch import nn
from torch import distributions as D
from callbacks import callback_default, callback_moons_ed, callback_mnist

#%%

class ownlinear(nn.Module):
    def __init__(self, size):
        super(ownlinear, self).__init__()
        self.a = nn.Parameter(torch.rand(size,))
        self.b = nn.Parameter(torch.rand(size,))
        
    def forward(self, x):
        return nn.functional.softplus(self.a) * x + nn.functional.softplus(self.b)

#%%
class VAE_experimental(nn.Module):
    def __init__(self, lr):
        super(VAE_experimental, self).__init__()
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
        x_mu = self.dec_mu(z)
        z_hat = z
        for _ in range(1):
            x_hat = self.dec_mu(z_hat)
            z_hat = self.enc_mu(x_hat)
        diff = (z-z_hat).norm(dim=-1, keepdim=True)
        x_std = self.switch*self.dec_std(diff) + (1-self.switch)*torch.tensor(0.02**2)
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
        x_mu = self.dec_mu(z)
        if self.switch:
            z_hat = z
            for _ in range(1):
                x_hat = self.dec_mu(z_hat)
                z_hat = self.enc_mu(x_hat)
            diff = (z-z_hat).norm(dim=2, keepdim=True).mean(dim=0, keepdim=True)
            x_std = self.switch*self.dec_std(diff) + (1-self.switch)*torch.tensor(0.02**2)
        else:
            x_std = (0.02**2)*torch.ones_like(x_mu)
        p_dist = D.Independent(D.Normal(x_mu, x_std+epsilon), 1)
        
        # Calculate loss
        prior = D.Independent(D.Normal(torch.zeros_like(z),
                                       torch.ones_like(z)), 1)
        log_px = p_dist.log_prob(x)
        kl = q_dist.log_prob(z) - prior.log_prob(z)
        elbo = (log_px - beta*kl).mean()
        iw_elbo = elbo.logsumexp(dim=0) - torch.tensor(float(iw_samples)).log()
        
        return iw_elbo.mean(), log_px.mean(), kl.mean(), x_mu, x_std, z, z_mu, z_std

#%%
class VAE_experimental_moons(VAE_experimental):
    def __init__(self, lr):
        super(VAE_experimental_moons, self).__init__(lr=lr)
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
        self.dec_std = nn.Sequential(ownlinear(2))
        
        self.callback = callback_moons_ed()
        self.latent_dim = 2
        
#%%
class VAE_experimental_mnist(VAE_experimental):
    def __init__(self, lr):
        super(VAE_experimental_mnist, self).__init__(lr=lr)
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
                                    nn.Linear(256, 784))
        self.dec_std = nn.Sequential(ownlinear(784))
        self.callback = callback_mnist()
        self.latent_dim = 2
        