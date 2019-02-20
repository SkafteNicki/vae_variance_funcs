# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 11:12:45 2019

@author: nsde
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

#%%
class callback_default(object):
    def __init__(self):
        pass
    
    def update(self, X, model, device, labels=None):
        pass
    
    def write(self, writer, epoch, label='cb'):
        pass
    
#%%
class callback_moons(callback_default):
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
            recon[i*100:(i+1)*100] = x_mu[0].cpu().numpy()
            
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
        
        # Make figures
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
        super(callback_moons_ed, self).__init__()
        self.fig6 = plt.figure()
        self.fig7 = plt.figure()
        self.fig8 = plt.figure()
        self.fig9 = plt.figure()

        
    def update(self, X, model, device, labels=None):
        super(callback_moons_ed, self).update(X, model, device, labels)

        z = np.stack([array.flatten() for array in np.meshgrid(
                      np.linspace(-5, 5, 100),
                      np.linspace(-5, 5, 100))]).T
        z = torch.tensor(z).to(torch.float32).to(device)
        x_mu, _ = model.decoder(z)
        z_hat, _ = model.encoder(x_mu)
        diff1 = (z-z_hat).norm(dim=-1, keepdim=True).cpu().numpy()
        vec = (z-z_hat).cpu().numpy() # vector field for quiver
        for _ in range(9):
            x_mu, _ = model.decoder(z_hat)
            z_hat, _ = model.encoder(x_mu)
        diff10 = (z-z_hat).norm(dim=-1, keepdim=True).cpu().numpy()
        for _ in range(90):
            x_mu, _ = model.decoder(z_hat)
            z_hat, _ = model.encoder(x_mu)
        diff100 = (z-z_hat).norm(dim=-1, keepdim=True).cpu().numpy()
        
        z = z.cpu().numpy()
        
        # Clear plots
        self.fig6.clear(); self.ax6 = self.fig6.add_subplot(111)
        self.fig7.clear(); self.ax7 = self.fig7.add_subplot(111)
        self.fig8.clear(); self.ax8 = self.fig8.add_subplot(111)
        self.fig9.clear(); self.ax9 = self.fig9.add_subplot(111)

        # Make figures
        cont = self.ax6.contourf(z[:,0].reshape(100, 100), z[:,1].reshape(100,100),
                                 diff1.reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax6)
        
        cont = self.ax7.contourf(z[:,0].reshape(100, 100), z[:,1].reshape(100,100),
                                 diff10.reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax7)
        cont = self.ax8.contourf(z[:,0].reshape(100, 100), z[:,1].reshape(100,100),
                                 diff100.reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax8)

        self.ax9.quiver(z[::4,0].reshape(50, 50), z[::4,1].reshape(50, 50),
                         vec[::4,0].reshape(50, 50), vec[::4,1].reshape(50, 50))
        
    def write(self, writer, epoch, label='cb'):
        super(callback_moons_ed, self).write(writer, epoch, label)
        writer.add_figure(label + '/diff1', self.fig6, epoch)
        writer.add_figure(label + '/diff10', self.fig7, epoch)
        writer.add_figure(label + '/diff100', self.fig8, epoch)
        writer.add_figure(label + '/quiver1', self.fig9, epoch)

#%%
class callback_mnist(callback_default):
    def __init__(self):
        self.fig1 = plt.figure()
        self.fig2 = plt.figure()
        self.fig3 = plt.figure()
        self.fig4 = plt.figure()
        self.fig5 = plt.figure()
    
    def update(self, X, model, device, labels=None):
        # Calculate some reconstructions and samples
        n = 10
        x = X[:n].to(device)
        _, _, _, x_mu, _, _, _, _ = model(x, 1.0, 1)
        self.img_recon = make_grid(torch.cat([x.reshape(-1, 1, 28, 28), 
            x_mu[0].reshape(-1, 1, 28, 28)]).cpu(), nrow=n).clamp(0,1)
        self.samples = make_grid(model.sample(n*n).cpu().reshape(-1, 1, 28, 28),
            nrow=n).clamp(0,1)
        
        # Extract latent coordinates
        N = X.shape[0]
        n_batch = int(np.ceil(N/100))
        latent = np.zeros((N, 2))
        for i in range(n_batch):
            x = X[i*100:(i+1)*100].to(device)
            z_mu, _ = model.encoder(x)
            latent[i*100:(i+1)*100] = z_mu.cpu().numpy()
        
        # Make diff and quiver plots
        z = np.stack([array.flatten() for array in np.meshgrid(
                      np.linspace(-5, 5, 100),
                      np.linspace(-5, 5, 100))]).T
        z = torch.tensor(z).to(torch.float32).to(device)
        x_mu, _ = model.decoder(z)
        z_hat, _ = model.encoder(x_mu)
        diff1 = (z-z_hat).norm(dim=-1, keepdim=True).cpu().numpy()
        vec = (z-z_hat).cpu().numpy() # vector field for quiver
        for _ in range(9):
            x_mu, _ = model.decoder(z_hat)
            z_hat, _ = model.encoder(x_mu)
        diff10 = (z-z_hat).norm(dim=-1, keepdim=True).cpu().numpy()
        z = z.cpu().numpy()
        
        # Forward grid to calculate meshgrid and std's
        grid = np.stack([array.flatten() for array in np.meshgrid(
                            np.linspace(-5, 5, 100),
                            np.linspace(-5, 5, 100))]).T
        x_mu, x_std = model.decoder(torch.tensor(grid).to(torch.float32).to(device))
        self.mesh = make_grid(x_mu[::4].cpu().reshape(-1, 1, 28, 28), nrow=50).clamp(0,1)
        x_std = x_std.cpu().numpy()
        
        # Clear plots
        self.fig1.clear(); self.ax1 = self.fig1.add_subplot(111)
        self.fig2.clear(); self.ax2 = self.fig2.add_subplot(111)
        self.fig3.clear(); self.ax3 = self.fig3.add_subplot(111)
        self.fig4.clear(); self.ax4 = self.fig4.add_subplot(111)
        self.fig5.clear(); self.ax5 = self.fig5.add_subplot(111)
        
        # Make figures
        scat = self.ax1.scatter(latent[:,0], latent[:,1], c=labels.numpy())
        plt.colorbar(scat, ax=self.ax1)
        
        cont = self.ax2.contourf(grid[:,0].reshape(100, 100),
                                 grid[:,1].reshape(100, 100),
                                 x_std.sum(axis=1).reshape(100, 100), 50) 
        plt.colorbar(cont, ax=self.ax2)
        
        cont = self.ax3.contourf(z[:,0].reshape(100, 100), z[:,1].reshape(100,100),
                                 diff1.reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax3)
        cont = self.ax4.contourf(z[:,0].reshape(100, 100), z[:,1].reshape(100,100),
                                 diff10.reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax4)
        self.ax5.quiver(z[::4,0].reshape(50, 50), z[::4,1].reshape(50, 50),
                        vec[::4,0].reshape(50, 50), vec[::4,1].reshape(50, 50))
        
        
    def write(self, writer, epoch, label='cb'):
        writer.add_image(label + '/recon', self.img_recon, epoch)
        writer.add_image(label + '/samples', self.samples, epoch)
        writer.add_image(label + '/meshgrid', self.mesh, epoch)
        writer.add_figure(label + '/latent', self.fig1, epoch)
        writer.add_figure(label + '/decoder_std', self.fig2, epoch)
        writer.add_figure(label + '/diff1', self.fig3, epoch)
        writer.add_figure(label + '/diff10', self.fig4, epoch)
        writer.add_figure(label + '/quiver', self.fig5, epoch)