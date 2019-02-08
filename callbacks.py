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
class callback_default(object):
    def __init__(self):
        pass
    
    def update(self, X, model, device, labels=None):
        pass
    
    def write(self, writer, epoch, label='cb'):
        pass
    
#%%
class callback_moons(object):
    def __init__(self):
        # Create figure and axis handles
        self.fig1 = plt.figure()
        self.fig2 = plt.figure()
        self.fig3 = plt.figure()
        self.fig4 = plt.figure()
        self.fig5 = plt.figure()
        
    def update(self, X, model, device, labels=None):
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
        self.fig1.clear(); self.ax1 = self.fig1.add_subplot(111) 
        self.fig2.clear(); self.ax2 = self.fig2.add_subplot(111) 
        self.fig3.clear(); self.ax3 = self.fig3.add_subplot(111) 
        self.fig4.clear(); self.ax4 = self.fig4.add_subplot(111) 
        self.fig5.clear(); self.ax5 = self.fig5.add_subplot(111) 
        
        # Data plot
        self.ax1.scatter(X[:,0].numpy(), X[:,1].numpy(), c=labels.numpy())
        self.ax2.scatter(recon[:,0], recon[:,1], c=labels.numpy())
        self.ax3.scatter(latent[:,0], latent[:,1], c=labels.numpy())
        cont = self.ax4.contourf(grid[:,0].reshape(100, 100),
                                 grid[:,1].reshape(100, 100),
                                 z_std.sum(axis=1).reshape(100, 100), 50) 
        plt.colorbar(cont, ax=self.ax4)
        cont = self.ax5.contourf(grid2[:,0].reshape(100, 100),
                                 grid2[:,1].reshape(100, 100),
                                 x_std.sum(axis=1).reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax5)
        
    def write(self, writer, epoch, label='cb'):
        # Write to tensorboard
        writer.add_figure(label+'/data', self.fig1, epoch)
        writer.add_figure(label+'/reconstruction', self.fig2, epoch)
        writer.add_figure(label+'/latent_space', self.fig3, epoch)
        writer.add_figure(label+'/encoder_std', self.fig4, epoch)
        writer.add_figure(label+'/decoder_std', self.fig5, epoch)

#%%      
class callback_moons_rbf(callback_moons):
    def update(self, X, model, device, labels=None):
        super(callback_moons_rbf, self).update(X, model, device, labels)
        # update figure 3 with clusters
        C = model.C.cpu().numpy()
        self.ax3.plot(C[:,0], C[:,1], 'g*')
        
#%%
class callback_moons_ed(callback_moons):
    def __init__(self):
        super(self, callback_moons_ed).__init__()
        self.fig6 = plt.figure()
        self.fig7 = plt.figure()
        self.fig8 = plt.figure()
        self.fig9 = plt.figure()
        
    def update(self, X, model, device, labels=None):
        super(self, callback_moons_ed).update(X, model, device, labels)
        N = X.shape[0]
        n_batch = int(np.ceil(N/100))
        for i in range(n_batch):
            x = X[i:100:(i+1)*100].to(device)
            _, _, _, x_mu, _, z, _, _ = model(x, 1.0, 1)
            z_hat, _ = model.encoder(x_mu)
            diff1 = (z-z_hat).norm(dim=1, keepdim=True)
            x_mu, _ = model.decoder(z_hat)
            z_hat, _ = model.encoder(x_mu)
            diff2 = (z-z_hat).norm(dim=1, keepdim=True)
            x_mu, _ = model.decoder(z_hat)
            z_hat, _ = model.encoder(x_mu)
            diff3 = (z-z_hat).norm(dim=1, keepdim=True)
            x_mu, _ = model.decoder(z_hat)
            z_hat, _ = model.encoder(x_mu)
            diff4 = (z-z_hat).norm(dim=1, keepdim=True)
        
        self.fig6.clear(); self.ax6 = self.fig6.add_subplot(111)
        self.fig6.clear(); self.ax6 = self.fig6.add_subplot(111)
        self.fig6.clear(); self.ax6 = self.fig6.add_subplot(111)
        self.fig6.clear(); self.ax6 = self.fig6.add_subplot(111)
        
        
    def write(self, writer, epoch, label='cb'):
        super(self, callback_moons_ed).write(writer, epoch, label)
        writer.add_figure(label + '/diff', self.fig6, epoch)
        