# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 09:45:16 2019

@author: nsde
"""
#%%
import torch
from tqdm import tqdm
import time, os, datetime
from tensorboardX import SummaryWriter
import numpy as np

#%%
class Trainer(object):
    def __init__(self, input_shape, model, use_cuda=True):
        self.model = model
        self.input_shape = input_shape
        self.use_cuda = use_cuda
        
        # Get the device and move model to gpu (if avaible)
        if torch.cuda.is_available() and self.use_cuda:
            self.device = torch.device('cuda')
            self.model.cuda()
        else:
            self.device = torch.device('cpu')
        
    def fit(self, Xtrain, n_epochs=10, warmup=1, batch_size=100, logdir=None, 
            iw_samples=1, beta=1.0, eval_epoch=10000, ytrain=None, 
            Xtest=None, ytest=None, log_epoch=100):
        # Print stats
        Ntrain = Xtrain.shape[0]
        print('Number of training points: ', Ntrain)
        n_batch_train = int(np.ceil(Ntrain/batch_size))
        if Xtest is not None: 
            Ntest = Xtest.shape[0]
            print('Number of test points:     ', Ntest)
            n_batch_test = int(np.ceil(Ntest/batch_size))
        
        # Dir to log results
        logdir = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') if logdir is None else logdir
        print('Logdir: ', logdir)
        if not os.path.exists(logdir): os.makedirs(logdir)
        
        # Summary writer
        writer = SummaryWriter(log_dir=logdir)
        
        # Main loop
        start = time.time()
        self.model.init_optim()
        for epoch in range(1, n_epochs+1):
            progress_bar = tqdm(desc='Epoch ' + str(epoch) + '/' + str(n_epochs), 
                                total=Ntrain, unit='samples')
            # Training loop
            self.model.train()
            train_loss = 0
            # Set switch for model
            self.model.switch = 1.0 if epoch > n_epochs/2 else 0.0
            for idx in range(n_batch_train):
                # Zero gradient
                self.model.zero_grad()
                
                # Extract batch
                data = Xtrain[batch_size*idx:batch_size*(idx+1)]
                data = data.reshape(-1, *self.input_shape).to(self.device)
                
                # Feed forward
                elbo, log_px, kl, x_mu, x_std, z, z_mu, z_std  = self.model(data, beta, iw_samples)
                train_loss += float(elbo.item())
                
                # Backprop
                (-elbo).backward()
                self.model.step()
                
                # Write to consol
                progress_bar.update(data.shape[0])
                progress_bar.set_postfix({'loss': elbo.item()})
                
                # Save to tensorboard
                iteration = epoch * n_batch_train + idx
                writer.add_scalar('train/total_loss', elbo, iteration)
                writer.add_scalar('train/log_px', log_px, iteration)
                writer.add_scalar('train/kl', kl, iteration)
                 
            progress_bar.set_postfix({'Average loss': train_loss/n_batch_train})
            progress_bar.close()
            
            # Evaluation loop
            self.model.eval()
            with torch.no_grad():
                if Xtest is not None:    
                    test_elbo, test_logpx, test_kl = 0, 0, 0
                    for idx in range(n_batch_test):
                        # Extract batch
                        data = Xtest[batch_size*idx:batch_size*(idx+1)]
                        data = data.reshape(-1, *self.input_shape).to(self.device)
                        
                        # Calculate loss
                        elbo, log_px, kl,_,_,_,_,_ = self.model(data, beta, iw_samples)
                        test_elbo += float(elbo.item())
                        test_logpx += float(log_px.item())
                        test_kl += float(kl.item())
                    writer.add_scalar('test/total_loss', test_elbo, iteration)
                    writer.add_scalar('test/log_px', test_logpx, iteration)
                    writer.add_scalar('test/kl', test_kl, iteration)
            
                # Callback (for logging)
                if epoch % log_epoch == 0 or epoch == n_epochs:
                    # Training set
                    self.model.callback.update(Xtrain, self.model, self.device, labels=ytrain)
                    self.model.callback.write(writer, epoch, 'train')
                    if Xtest is not None:
                        # Test set
                        self.model.callback.update(Xtest, self.model, self.device, labels=ytest)
                        self.model.callback.write(writer, epoch, 'train')
            
        print('Total train time:', time.time() - start)