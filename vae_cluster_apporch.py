#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 08:30:20 2019

@author: nsde
"""

#%%
from data import two_moons
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.distributions as D
#%%
class args:
    n_clust = 2
    lr = 0.001
    batch_size = 2000
    n = 1000
    n_epochs = 5000
    warmup = 1000

#%%
if __name__ == '__main__':
    # Load dataset
    X, y = two_moons(1000)
    
    # Preprocessing
    clusterAlg = AgglomerativeClustering(n_clusters=args.n_clust, linkage='single')
    clusterAlg.fit(X)
    labels = torch.tensor(clusterAlg.labels_)

    class newvae(nn.Module):
        def __init__(self, ):
            super(newvae, self).__init__()
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
            self.cluster = nn.Sequential(nn.Linear(2, 100),
                                         nn.ReLU(),
                                         nn.Linear(100, 100),
                                         nn.ReLU(),
                                         nn.Linear(100, args.n_clust),
                                         nn.Softmax())
            self.cluster_loss = nn.NLLLoss(reduction='sum')
            
        def encoder(self, x):
            return self.enc_mu(x), self.enc_std(x)
            
        def decoder(self, z, switch):
            x_mu = self.dec_mu(z)
            prop = self.cluster(z)
            x_std = -(prop * prop.log()).sum(dim=1, keepdim=True)
            return x_mu, switch*x_std + (1-switch)*torch.tensor(0.02**2)
            
        def forward(self, x, switch, labels):
            z_mu, z_std = self.encoder(x)
            q_dist = D.Independent(D.Normal(z_mu, z_std), 1)
            z = q_dist.rsample()
            
            x_mu, x_std = self.decoder(z, switch)
            if switch:
                prop = self.cluster(z)
                print(prop[:5], labels[:5])
                print(prop[-5:], labels[-5:])
                nll_loss = self.cluster_loss(prop.log(), labels)
            else:
                nll_loss = 0.0
            p_dist = D.Independent(D.Normal(x_mu, x_std), 1)
            
            prior = D.Independent(D.Normal(torch.zeros_like(z),
                                           torch.ones_like(z)), 1)
            log_px = p_dist.log_prob(x)
            kl = q_dist.log_prob(z) - prior.log_prob(z)
            elbo = (log_px - 1.0*kl).mean()
        
            return elbo.mean() - nll_loss, log_px.mean(), kl.mean(), x_mu, x_std, z, z_mu, z_std
    
    model = newvae()
    
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
            switch = 1.0 if 1000 < e else 0.0 
            # Forward pass
            x = X[i*args.batch_size:(i+1)*args.batch_size].to(device)
            y = labels[i*args.batch_size:(i+1)*args.batch_size].to(device)
            elbo, recon, kl, x_mu, x_std, z, z_mu, z_std = model(x, switch, y)
            
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

                _, x_std = model.decoder(torch.tensor(grid).to(torch.float32).to(device), switch)              
                
                x_std = x_std.cpu().numpy()
                for coll in cont2.collections: ax[1,2].collections.remove(coll)
                cont2 = ax[1,2].contourf(grid[:,0].reshape(100, 100),
                                         grid[:,1].reshape(100, 100),
                                         np.log(x_std.sum(axis=1)).reshape(100, 100))

                if e == args.n_epochs: # put colorbars on plot in the end
                    plt.colorbar(cont1, ax=ax[0,2])
                    plt.colorbar(cont2, ax=ax[1,2])
                
                # Draw
                plt.draw()
                plt.pause(0.01)
    
    plt.show(block=True)