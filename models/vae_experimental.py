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

import matplotlib.pyplot as plt
import numpy as np

class ownlinear(nn.Module):
    def __init__(self,):
        super(ownlinear, self).__init__()
        self.a = nn.Parameter(torch.tensor(np.random.rand(2,).astype(np.float32)))
        self.b = nn.Parameter(torch.tensor(np.random.rand(2,).astype(np.float32)))
        
    def forward(self, x):
        return nn.functional.softplus(self.a) * x + nn.functional.softplus(self.b)

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
        self.dec_std = nn.Sequential(ownlinear())
        
        # For plotting
        self.fig, self.ax = plt.subplots(2, 3)
        self.cont1 = self.ax[0,0].contourf(np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))
        self.ax[0,0].set_title('z - zxz(1)')
        self.cont2 = self.ax[0,1].contourf(np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))
        self.ax[0,1].set_title('z - zxz(2)')
        self.cont3 = self.ax[0,2].contourf(np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))
        self.ax[0,2].set_title('z - zxz(3)')
        self.cont4 = self.ax[1,0].contourf(np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))
        self.ax[1,0].set_title('z - zxz(4)')
        self.cont5 = self.ax[1,1].contourf(np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))
        self.ax[1,1].set_title('z - zxz(5)')
        self.cont6 = self.ax[1,2].contourf(np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))
        self.ax[1,2].set_title('z - zxz(6)')
        
    def encoder(self, x):
        return self.enc_mu(x), self.enc_std(x)
        
    def decoder(self, z, switch=1.0):
        x_mu = self.dec_mu(z)
        z_hat = self.enc_mu(x_mu)
        diff = (z-z_hat).norm(dim=1, keepdim=True)
        x_std = switch*self.dec_std(diff) + (1-switch)*torch.tensor(0.02**2)
        
        if self.training == False:
            
            p1 = (z-z_hat).norm(dim=1, keepdim=True).cpu().numpy()
            x_mu = self.dec_mu(z_hat)
            z_hat = self.enc_mu(x_mu)
            p2 = (z-z_hat).norm(dim=1, keepdim=True).cpu().numpy()
            x_mu = self.dec_mu(z_hat)
            z_hat = self.enc_mu(x_mu)
            p3 = (z-z_hat).norm(dim=1, keepdim=True).cpu().numpy()
            x_mu = self.dec_mu(z_hat)
            z_hat = self.enc_mu(x_mu)
            p4 = (z-z_hat).norm(dim=1, keepdim=True).cpu().numpy()
            x_mu = self.dec_mu(z_hat)
            z_hat = self.enc_mu(x_mu)
            p5 = (z-z_hat).norm(dim=1, keepdim=True).cpu().numpy()
            x_mu = self.dec_mu(z_hat)
            z_hat = self.enc_mu(x_mu)
            p6 = (z-z_hat).norm(dim=1, keepdim=True).cpu().numpy()
            
            z = z.cpu().numpy()
            
            for coll in self.cont1.collections: self.ax[0,0].collections.remove(coll)
            self.cont1 = self.ax[0,0].contourf(z[:,0].reshape(100, 100),
                                             z[:,1].reshape(100, 100),
                                             np.log(p1.sum(axis=1)).reshape(100, 100))
            for coll in self.cont2.collections: self.ax[0,1].collections.remove(coll)
            self.cont2 = self.ax[0,1].contourf(z[:,0].reshape(100, 100),
                                             z[:,1].reshape(100, 100),
                                             np.log(p2.sum(axis=1)).reshape(100, 100))
            for coll in self.cont3.collections: self.ax[0,2].collections.remove(coll)
            self.cont3 = self.ax[0,2].contourf(z[:,0].reshape(100, 100),
                                             z[:,1].reshape(100, 100),
                                             np.log(p3.sum(axis=1)).reshape(100, 100))
            for coll in self.cont4.collections: self.ax[1,0].collections.remove(coll)
            self.cont4 = self.ax[1,0].contourf(z[:,0].reshape(100, 100),
                                             z[:,1].reshape(100, 100),
                                             np.log(p4.sum(axis=1)).reshape(100, 100))
            for coll in self.cont5.collections: self.ax[1,1].collections.remove(coll)
            self.cont5 = self.ax[1,1].contourf(z[:,0].reshape(100, 100),
                                             z[:,1].reshape(100, 100),
                                             np.log(p5.sum(axis=1)).reshape(100, 100))
            for coll in self.cont6.collections: self.ax[1,2].collections.remove(coll)
            self.cont6 = self.ax[1,2].contourf(z[:,0].reshape(100, 100),
                                             z[:,1].reshape(100, 100),
                                             np.log(p6.sum(axis=1)).reshape(100, 100))
            self.fig.canvas.draw()
            plt.pause(0.01)
            
            print(list(self.dec_std.parameters()))
        
        return x_mu, x_std
    
    def forward(self, x, beta=1.0, switch=1.0, iw_samples=1):
        # Encoder step
        z_mu, z_std = self.encoder(x)
        q_dist = D.Independent(D.Normal(z_mu, z_std), 1)
        z = q_dist.rsample()
        
        # Decoder step
        x_mu, x_std = self.decoder(z, switch)
        
        p_dist = D.Independent(D.Normal(x_mu, x_std), 1)
        
        # Calculate loss
        prior = D.Independent(D.Normal(torch.zeros_like(z),
                                       torch.ones_like(z)), 1)
        log_px = p_dist.log_prob(x)
        kl = q_dist.log_prob(z) - prior.log_prob(z)
        elbo = (log_px - beta*kl).mean()
        
        return elbo.mean(), log_px.mean(), kl.mean(), x_mu, x_std, z, z_mu, z_std
    