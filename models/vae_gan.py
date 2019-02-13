# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 11:02:44 2019

@author: nsde
"""
#%%
import torch
from torch import nn
from torch import distributions as D 
from torch.nn import functional as F
from callbacks import callback_default, callback_moons
from itertools import chain

class ownlinear(nn.Module):
    def __init__(self,):
        super(ownlinear, self).__init__()
        self.a = nn.Parameter(torch.rand(2,))
        self.b = nn.Parameter(torch.rand(2,))
        
    def forward(self, x):
        return nn.functional.softplus(self.a) * x + nn.functional.softplus(self.b)

#%%
class VAE_gan_base(nn.Module):
    def __init__(self, lr):
        super(VAE_gan_base, self).__init__()
        self.switch = 0.0
        self.lr = lr
        self.callback = callback_default()
        
    def init_optim(self):
        self.optimizer1 = torch.optim.Adam(chain(self.enc_mu.parameters(),
                                                 self.enc_std.parameters(),
                                                 self.dec_mu.parameters()), lr=self.lr)
        self.optimizer2 = torch.optim.SGD(chain(self.adverserial.parameters(),
                                                 self.dec_std.parameters()), lr=self.lr)
        
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
        prop = self.adverserial(x_mu)
        return x_mu, self.switch*self.dec_std(prop)+(1-self.switch)*torch.tensor(0.02**2)
    
    def forward(self, x, beta=1.0, iw_samples=1):
        # Encoder step
        z_mu, z_std = self.encoder(x)
        q_dist = D.Independent(D.Normal(z_mu, z_std), 1)
        z = q_dist.rsample([iw_samples])
        
        # Decoder step
        x_mu, x_std = self.decoder(z)
        
        if self.switch:
            valid = -0.3*torch.rand((x.shape[0], 1), device = x.device) + 0.3
            fake = -0.3*torch.rand((x.shape[0], 1), device = x.device)+ 1.0
            labels = torch.cat([valid, fake], dim=0)
            x_cat = torch.cat([x, x_mu[0]], dim=0)
            prop = self.adverserial(x_cat)
            advert_loss = F.binary_cross_entropy(prop, labels, reduction='sum')
            x_std = self.dec_std(prop)
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
        
        return iw_elbo.mean() - advert_loss, log_px.mean(), kl.mean(), x_mu, x_std, z, z_mu, z_std
    
#%%
class VAE_gan_moons(VAE_gan_base):
    def __init__(self, lr):
        super(VAE_gan_moons, self).__init__(lr=lr)
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
        self.adverserial = nn.Sequential(nn.Linear(2, 100),
                                         nn.LeakyReLU(),
                                         nn.Linear(100, 100),
                                         nn.LeakyReLU(),
                                         nn.Linear(100, 1),
                                         nn.Sigmoid())
        self.dec_std = nn.Sequential(ownlinear())
        
        self.callback = callback_moons()