#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:30:36 2019

@author: nsde
"""

#%%
import argparse, datetime
import torch
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
    ms.add_argument('--model', type=str, default='vae_rbf', help='model to train')
    ms.add_argument('--beta', type=float, default=1.0, help='weighting of KL term')
    ms.add_argument('--switch', type=lambda x: (str(x).lower() == 'true'), default=True, help='use switch for variance')
    # Training settings
    ts = parser.add_argument_group('Training settings')
    ts.add_argument('--n_epochs', type=int, default=3000, help='number of epochs of training')
    ts.add_argument('--batch_size', type=int, default=2000, help='size of the batches')
    ts.add_argument('--warmup', type=int, default=1000, help='number of warmup epochs for kl-terms')
    ts.add_argument('--lr', type=float, default=1e-3, help='learning rate for adam optimizer')

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

    # Logdir for results
    if args.logdir == '':
        logdir = 'res/' + args.model + '/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    else:
        logdir = 'res/' + args.model + '/' + args.logdir
    
    # Load dataset
    X, y = two_moons(args.n)
    
    # Construct models
    model_class = get_model(args.model)
    model = model_class()
    
    if torch.cuda.is_available():
        model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    fig, ax = plt.subplots(2, 3)
    ax[0,0].plot(X[:args.n,0].numpy(), X[:args.n,1].numpy(), 'r.')
    ax[0,0].plot(X[args.n:,0].numpy(), X[args.n:,1].numpy(), 'b.')
    ax[0,0].set_xlim(-2,2)
    ax[0,0].set_ylim(-2,2)
    ax[0,0].set_title('Training data')
    line, = ax[1,0].semilogy([ ], 'b-')
    ax[0,1].set_title('Negative ELBO')
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
    
    losslist = [ ]
    n_batch = int(np.ceil(X.shape[0] // args.batch_size))
    for e in range(args.n_epochs):
        loss = 0
        for i in range(n_batch):
            optimizer.zero_grad()
            
            # Training params
            if args.switch:
                switch = 1.0 if args.n_epochs/2 < e else 0.0 
            else:
                switch = 1.0
            beta = args.beta*float(np.minimum(1, e/args.warmup))
            
            # Forward pass
            x = X[i*args.batch_size:(i+1)*args.batch_size].to(device)
            elbo, x_mu, x_var, z, z_mu, z_var = model(x, beta, switch)
            
            # Backward pass
            (-elbo).backward() # maximize elbo <-> minimize -elbo
            optimizer.step()
            print('Epoch: ', e, '/', args.n_epochs, ', elbo:', elbo.item(),
                    ', mean x_var:', x_var.mean().item())
            loss += -elbo.item()
        losslist.append(loss)
        
        if e % 50 == 0:
            with torch.no_grad():
                x_mu = x_mu.detach().cpu()
                z = z.detach().cpu()
                # Loss 
                ax[1,0].semilogy(losslist, 'b-')
                
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
                _, _, _, _, _, z_var = model(torch.tensor(grid).to(torch.float32).to(device), 
                                             beta, switch)
                z_var = z_var.cpu().numpy()
                for coll in cont1.collections: ax[0,2].collections.remove(coll)
                cont1 = ax[0,2].contourf(grid[:,0].reshape(100, 100),
                                         grid[:,1].reshape(100, 100),
                                         np.log(z_var.sum(axis=1)).reshape(100, 100))
                
                if args.model != 'vae_single':
                    # Decoder variance
                    grid = np.stack([array.flatten() for array in np.meshgrid(
                            np.linspace(-5, 5, 100),
                            np.linspace(-5, 5, 100))]).T
                    _, x_std = model.decoder(torch.tensor(grid).to(torch.float32).to(device), 
                                             switch)
                    x_std = x_std.cpu().numpy()
                    for coll in cont2.collections: ax[1,2].collections.remove(coll)
                    cont2 = ax[1,2].contourf(grid[:,0].reshape(100, 100),
                                             grid[:,1].reshape(100, 100),
                                             np.log(x_std.sum(axis=1)).reshape(100, 100))
                else:
                    print('x_var:', x_var)
                if args.model == 'vae_rbf':
                    scat5.set_data(model.C[:,0].detach().cpu(), model.C[:,1].detach().cpu())
                    
                # Draw
                plt.draw()
                plt.pause(0.01)