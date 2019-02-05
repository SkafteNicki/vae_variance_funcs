# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 11:12:45 2019

@author: nsde
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import torch

#%%
class callback_moons(object):
    def __init__(self):
        # Create figure and axis handles
        self.fig1, self.ax1 = plt.subplots()
        self.fig2, self.ax2 = plt.subplots()
        self.fig3, self.ax3 = plt.subplots()
        self.fig4, self.ax4 = plt.subplots()
        self.fig5, self.ax5 = plt.subplots()
        
    def __call__(self, X, model, writer, device, epoch, label='cb', labels=None):
        # Extract latent codes and reconstrutions
        N = X.shape[0]
        n_batch = int(np.ceil(N/100))
        latent = np.zeros((N, 2))
        recon = np.zeros((N,2))
        for i in range(n_batch):
            x = X[i*100:(i+1)*100].to(device)
            _, _, _, x_mu, _, _, z_mu, _ = model(x, 1.0, 1)
            latent[i*100:(i+1)*100] = z_mu.cpu().numpy()
            recon[i*100:(i+1)*100] = x_mu.cpu().numpy()
            
        # Forward grid to calculate std's
        grid = np.stack([array.flatten() for array in np.meshgrid(
                            np.linspace(-2, 2, 100),
                            np.linspace(-2, 2, 100))]).T
        _, z_std = model.encoder(torch.tensor(grid).to(torch.float32).to(device))
        z_std = z_std.cpu().numpy()
        
        grid2 = np.stack([array.flatten() for array in np.meshgrid(
                            np.linspace(-5, 5, 100),
                            np.linspace(-5, 5, 100))]).T
        _, x_std = model.decoder(torch.tensor(grid2).to(torch.float32).to(device))
        x_std = x_std.cpu().numpy()
        
        # Clear plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        
        # Data plot
        self.ax1.scatter(X[:,0].numpy(), X[:,1].numpy(), c=labels.numpy())
        self.ax2.scatter(recon[:,0], recon[:,1], c=labels.numpy())
        self.ax3.scatter(latent[:,0], latent[:,1], c=labels.numpy())
        self.ax4.contourf(grid[:,0].reshape(100, 100),
                          grid[:,1].reshape(100, 100),
                          z_std.sum(axis=1).reshape(100, 100), 50)
        self.ax5.contourf(grid2[:,0].reshape(100, 100),
                          grid2[:,1].reshape(100, 100),
                          x_std.sum(axis=1).reshape(100, 100), 50)
        
        # Write to tensorboard
        writer.add_figure(label+'/data', self.fig1, epoch)
        writer.add_figure(label+'/reconstruction', self.fig2, epoch)
        writer.add_figure(label+'/latent_space', self.fig3, epoch)
        writer.add_figure(label+'/encoder_std', self.fig4, epoch)
        writer.add_figure(label+'/decoder_std', self.fig5, epoch)