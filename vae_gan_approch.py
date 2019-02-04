#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:25:24 2019

@author: nsde
"""

#%%
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as D

import argparse, datetime
import torch
import matplotlib.pyplot as plt
import numpy as np

from models import get_model
from data import two_moons

from itertools import chain


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

#%%
def argparser():
    """ Argument parser for the main script """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Model settings
    ms = parser.add_argument_group('Model settings')
    ms.add_argument('--model', type=str, default='vae_experimental', help='model to train')
    ms.add_argument('--beta', type=float, default=1.0, help='weighting of KL term')
    ms.add_argument('--switch', type=lambda x: (str(x).lower() == 'true'), default=True, help='use switch for variance')
    ms.add_argument('--anneling', type=lambda x: (str(x).lower() == 'true'), default=True, help='use anneling for kl term')
    
    # Training settings
    ts = parser.add_argument_group('Training settings')
    ts.add_argument('--n_epochs', type=int, default=2000, help='number of epochs of training')
    ts.add_argument('--batch_size', type=int, default=2000, help='size of the batches')
    ts.add_argument('--warmup', type=int, default=1000, help='number of warmup epochs for kl-terms')
    ts.add_argument('--lr', type=float, default=1e-3, help='learning rate for adam optimizer')
    ts.add_argument('--iw_samples', type=int, default=1, help='number of importance weighted samples')

    # Dataset settings
    ds = parser.add_argument_group('Dataset settings')
    ds.add_argument('--n', type=int, default=1000, help='number of points in each class')
    ds.add_argument('--logdir', type=str, default='res', help='where to store results')
    ds.add_argument('--dataset', type=str, default='mnist', help='dataset to use')
    
    # Parse and return
    args = parser.parse_args()
    return args

#%%
if __name__ == '__main__':
    # Input arguments
    args = argparser()
    
    # Load dataset
    X, y = two_moons(args.n)
    
    model = VAE_experimental()
    
    if torch.cuda.is_available():
        model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Optimizer
    optimizer = torch.optim.Adam(chain(model.enc_mu.parameters(),
                                       model.enc_std.parameters(),
                                       model.dec_mu.parameters()), lr=args.lr)
    optimizer2 = torch.optim.Adam(model.adverserial.parameters(), lr=args.lr)
