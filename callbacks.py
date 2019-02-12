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
        self.fig10 = plt.figure()
        self.fig11 = plt.figure()
        self.fig12 = plt.figure()
        self.fig13 = plt.figure()
        self.fig14 = plt.figure()
        self.fig15 = plt.figure()
        self.fig16 = plt.figure()
        self.fig17 = plt.figure()
        
    def update(self, X, model, device, labels=None):
        super(callback_moons_ed, self).update(X, model, device, labels)

        z = np.stack([array.flatten() for array in np.meshgrid(
                      np.linspace(-5, 5, 100),
                      np.linspace(-5, 5, 100))]).T
        z = torch.tensor(z).to(torch.float32).to(device)
        x_mu, _ = model.decoder(z)
        z_hat, _ = model.encoder(x_mu)
        diff1 = (z-z_hat).norm(dim=-1, keepdim=True).cpu().numpy()
        vec = (z-z_hat).cpu().numpy() # vector field
        x_mu, _ = model.decoder(z_hat)
        z_hat, _ = model.encoder(x_mu)
        diff2 = (z-z_hat).norm(dim=-1, keepdim=True).cpu().numpy()
        x_mu, _ = model.decoder(z_hat)
        z_hat, _ = model.encoder(x_mu)
        diff3 = (z-z_hat).norm(dim=-1, keepdim=True).cpu().numpy()
        x_mu, _ = model.decoder(z_hat)
        z_hat, _ = model.encoder(x_mu)
        diff4 = (z-z_hat).norm(dim=-1, keepdim=True).cpu().numpy()
        x_mu, _ = model.decoder(z_hat)
        z_hat, _ = model.encoder(x_mu)
        diff5 = (z-z_hat).norm(dim=-1, keepdim=True).cpu().numpy()
        x_mu, _ = model.decoder(z_hat)
        z_hat, _ = model.encoder(x_mu)
        diff6 = (z-z_hat).norm(dim=-1, keepdim=True).cpu().numpy()
        x_mu, _ = model.decoder(z_hat)
        z_hat, _ = model.encoder(x_mu)
        diff7 = (z-z_hat).norm(dim=-1, keepdim=True).cpu().numpy()
        x_mu, _ = model.decoder(z_hat)
        z_hat, _ = model.encoder(x_mu)
        diff8 = (z-z_hat).norm(dim=-1, keepdim=True).cpu().numpy()
        x_mu, _ = model.decoder(z_hat)
        z_hat, _ = model.encoder(x_mu)
        diff9 = (z-z_hat).norm(dim=-1, keepdim=True).cpu().numpy()
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
        self.fig10.clear(); self.ax10 = self.fig10.add_subplot(111)
        self.fig11.clear(); self.ax11 = self.fig11.add_subplot(111)
        self.fig12.clear(); self.ax12 = self.fig12.add_subplot(111)
        self.fig13.clear(); self.ax13 = self.fig13.add_subplot(111)
        self.fig14.clear(); self.ax14 = self.fig14.add_subplot(111)
        self.fig15.clear(); self.ax15 = self.fig15.add_subplot(111)
        self.fig16.clear(); self.ax16 = self.fig16.add_subplot(111)
        self.fig17.clear(); self.ax17 = self.fig17.add_subplot(111)
        
        # Make figures
        cont = self.ax6.contourf(z[:,0].reshape(100, 100), z[:,1].reshape(100,100),
                                 diff1.reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax6)
        cont = self.ax7.contourf(z[:,0].reshape(100, 100), z[:,1].reshape(100,100),
                                 diff2.reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax7)
        cont = self.ax8.contourf(z[:,0].reshape(100, 100), z[:,1].reshape(100,100),
                                 diff3.reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax8)
        cont = self.ax9.contourf(z[:,0].reshape(100, 100), z[:,1].reshape(100,100),
                                 diff4.reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax9)
        cont = self.ax10.contourf(z[:,0].reshape(100, 100), z[:,1].reshape(100,100),
                                 diff5.reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax10)
        cont = self.ax11.contourf(z[:,0].reshape(100, 100), z[:,1].reshape(100,100),
                                 diff6.reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax11)
        cont = self.ax12.contourf(z[:,0].reshape(100, 100), z[:,1].reshape(100,100),
                                 diff7.reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax12)
        cont = self.ax13.contourf(z[:,0].reshape(100, 100), z[:,1].reshape(100,100),
                                 diff8.reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax13)
        cont = self.ax14.contourf(z[:,0].reshape(100, 100), z[:,1].reshape(100,100),
                                 diff9.reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax14)
        cont = self.ax15.contourf(z[:,0].reshape(100, 100), z[:,1].reshape(100,100),
                                 diff10.reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax15)
        cont = self.ax16.contourf(z[:,0].reshape(100, 100), z[:,1].reshape(100,100),
                                 diff100.reshape(100, 100), 50)
        plt.colorbar(cont, ax=self.ax16)

        self.ax17.quiver(z[::4,0].reshape(50, 50), z[::4,1].reshape(50, 50),
                         vec[::4,0].reshape(50, 50), vec[::4,1].reshape(50, 50))
        
    def write(self, writer, epoch, label='cb'):
        super(callback_moons_ed, self).write(writer, epoch, label)
        writer.add_figure(label + '/diff1', self.fig6, epoch)
        writer.add_figure(label + '/diff2', self.fig7, epoch)
        writer.add_figure(label + '/diff3', self.fig8, epoch)
        writer.add_figure(label + '/diff4', self.fig9, epoch)
        writer.add_figure(label + '/diff5', self.fig10, epoch)
        writer.add_figure(label + '/diff6', self.fig11, epoch)
        writer.add_figure(label + '/diff7', self.fig12, epoch)
        writer.add_figure(label + '/diff8', self.fig13, epoch)
        writer.add_figure(label + '/diff9', self.fig14, epoch)
        writer.add_figure(label + '/diff10', self.fig15, epoch)
        writer.add_figure(label + '/diff100', self.fig16, epoch)
        writer.add_figure(label + '/quiver1', self.fig17, epoch)
        