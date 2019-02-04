#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:26:16 2019

@author: nsde
"""

#%%
import argparse, datetime
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

from models import get_model
from data import two_moons

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

    
    # Parse and return
    args = parser.parse_args()
    return args

#%%
class mymodel(nn.Module):
    def __init__(self, ):
        super(mymodel, self).__init__()
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
        
    def encoder(self, x):
        return self.enc_mu, self.end_std
        
    def decoder(self, z, switch=1.0):
        x_mu = self.dec_mu(z)
        z_hat = self.end_mu(x_mu)
        x_std = switch*(z-z_hat).norm(dim=1, keepdim=1) + (1-switch)*torch.tensor(0.02**2)
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
        
#%%
if __name__ == '__main__':
    # Input arguments
    args = argparser()

    # Load dataset
    X, y = two_moons(args.n)
    
    # Construct models
    model = model_class()
    
    if torch.cuda.is_available():
        model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # For plotting
    fig, ax = plt.subplots(3, 3)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    ax[0,0].plot(X[:args.n,0].numpy(), X[:args.n,1].numpy(), 'r.')
    ax[0,0].plot(X[args.n:,0].numpy(), X[args.n:,1].numpy(), 'b.')
    ax[0,0].set_xlim(-2,2)
    ax[0,0].set_ylim(-2,2)
    ax[0,0].set_title('Training data')
    line, = ax[1,0].semilogy([ ], 'b-')
    ax[1,0].set_title('Negative ELBO')
    scat1, = ax[0,1].plot([ ], [ ], 'r.')
    scat2, = ax[0,1].plot([ ], [ ], 'b.')
    ax[0,1].set_title('Reconstruction')
    scat3, = ax[1,1].plot([ ], [ ], 'r.')
    scat4, = ax[1,1].plot([ ], [ ], 'b.')
    scat5, = ax[1,1].plot([ ], [ ], 'g*')
    ax[1,1].set_title('Latent space')
    cont1 = ax[0,2].contourf(np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))
    ax[0,2].set_title('Encoder variance')
    cont2 = ax[1,2].contourf(np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))
    ax[1,2].set_title('Decoder variance')
    line2, = ax[2,0].plot([ ], 'b-')
    ax[2,0].set_title('Reconstruction')
    line3, = ax[2,1].plot([ ], 'b-')
    ax[2,1].set_title('KL')
    line4, = ax[2,2].semilogy([ ], 'b-')
    ax[2,2].set_title('Mean decoder std estimate')
    
    losslist, losslist2, losslist3, mean_std = [ ], [ ], [ ], [ ]
    n_batch = int(np.ceil(X.shape[0] // args.batch_size))
    for e in range(1, args.n_epochs+1):
        loss, loss_recon, loss_kl = 0, 0, 0
        for i in range(n_batch):
            optimizer.zero_grad()
            
            # Training params
            if args.switch:
                switch = 1.0 if args.n_epochs/2 < e else 0.0 
            else:
                switch = 1.0
            if args.anneling:
                beta = args.beta*float(np.minimum(1, e/args.warmup))
            else:
                beta = args.beta

            # Forward pass
            x = X[i*args.batch_size:(i+1)*args.batch_size].to(device)
            elbo, recon, kl, x_mu, x_std, z, z_mu, z_std = model(x, beta, switch, args.iw_samples)
            
            # Backward pass
            (-elbo).backward() # maximize elbo <-> minimize -elbo
            optimizer.step()
            
            # Save
            loss += -elbo.item()
            loss_recon += recon.item()
            loss_kl += kl.item()
            
        # Print progress
        print('Epoch: {0}/{1}, ELBO: {2:.3f}, Recon: {3:.3f}, KL: {4:.3f}'.format(
                e, args.n_epochs, loss, loss_recon, loss_kl))
        losslist.append(loss)
        losslist2.append(abs(loss_recon))
        losslist3.append(abs(loss_kl))
        mean_std.append(x_std.mean().item())
        
        if e % 50 == 0:
            with torch.no_grad():
                x_mu = x_mu.detach().cpu()
                z = z.detach().cpu()
                # Loss 
                ax[1,0].semilogy(losslist, 'b-')
                ax[2,0].semilogy(losslist2, 'b-')
                ax[2,1].semilogy(losslist3, 'b-')
                ax[2,2].semilogy(mean_std, 'b-')
                
                # Reconstruction
                scat1.set_data(x_mu[:args.n,0], x_mu[:args.n,1])
                scat2.set_data(x_mu[args.n:,0], x_mu[args.n:,1])
                ax[0,1].set_xlim(-2,2)
                ax[0,1].set_ylim(-2,2)
    
                # Latent space
                scat3.set_data(z[:args.n,0], z[:args.n,1])
                scat4.set_data(z[args.n:,0], z[args.n:,1])
                ax[1,1].set_xlim(-5,5)
                ax[1,1].set_ylim(-5,5)
                
                # Encoder variance
                grid = np.stack([array.flatten() for array in np.meshgrid(
                        np.linspace(-2, 2, 100),
                        np.linspace(-2, 2, 100))]).T
                _, z_std = model.encoder(torch.tensor(grid).to(torch.float32).to(device))
                z_std = z_std.cpu().numpy()
                for coll in cont1.collections: ax[0,2].collections.remove(coll)
                cont1 = ax[0,2].contourf(grid[:,0].reshape(100, 100),
                                         grid[:,1].reshape(100, 100),
                                         np.log(z_std.sum(axis=1)).reshape(100, 100))
                
                # Decoder variance
                grid = np.stack([array.flatten() for array in np.meshgrid(
                        np.linspace(-5, 5, 100),
                        np.linspace(-5, 5, 100))]).T
                if args.model != 'vae_student':
                    _, x_std = model.decoder(torch.tensor(grid).to(torch.float32).to(device), 
                                             switch)
                else:
                    _, x_df, x_scale = model.decoder(torch.tensor(grid).to(torch.float32).to(device), 
                                             switch)
                    x_std = x_df / (x_df - 2) * x_scale
                
                x_std = x_std.cpu().numpy()
                for coll in cont2.collections: ax[1,2].collections.remove(coll)
                cont2 = ax[1,2].contourf(grid[:,0].reshape(100, 100),
                                         grid[:,1].reshape(100, 100),
                                         np.log(x_std.sum(axis=1)).reshape(100, 100))
                
                if hasattr(model, 'C'): # plot clusters
                    scat5.set_data(model.C[:,0].detach().cpu(), model.C[:,1].detach().cpu())
                
                if e == args.n_epochs: # put colorbars on plot in the end
                    plt.colorbar(cont1, ax=ax[0,2])
                    plt.colorbar(cont2, ax=ax[1,2])
                
                # Draw
                plt.draw()
                plt.pause(0.01)
                
    plt.savefig(str(args.model) + '.pdf')
    plt.show(block=True)