#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:30:36 2019

@author: nsde
"""

#%%
import argparse, datetime

from models import get_model
from data import two_moons, mnist
from training import Trainer

#%%
def argparser():
    """ Argument parser for the main script """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Model settings
    ms = parser.add_argument_group('Model settings')
    ms.add_argument('--model', type=str, default='vae_full', help='model to train')
    ms.add_argument('--beta', type=float, default=1.0, help='weighting of KL term')
    
    # Training settings
    ts = parser.add_argument_group('Training settings')
    ts.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
    ts.add_argument('--batch_size', type=int, default=100, help='size of the batches')
    ts.add_argument('--warmup', type=int, default=1, help='number of warmup epochs for kl-terms')
    ts.add_argument('--lr', type=float, default=1e-3, help='learning rate for adam optimizer')
    ts.add_argument('--iw_samples', type=int, default=1, help='number of importance weighted samples')
    ts.add_argument('--log_epoch', type=int, default=100, help='how many epochs to pass before calling the callback')
    ts.add_argument('--switch_epoch', type=int, default=None, help='when to switch on variance network, default n_epochs/2')
    
    # Dataset settings
    ds = parser.add_argument_group('Dataset settings')
    ds.add_argument('--n', type=int, default=1000, help='number of points in each class')
    ds.add_argument('--logdir', type=str, default='', help='where to store results')
    ds.add_argument('--dataset', type=str, default='moons', help='dataset to use')
    
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
    if args.dataset == 'moons':
        Xtrain, ytrain = two_moons(args.n, train=True)
        Xtest, ytest = two_moons(int(args.n/2), train=False)
        input_shape = (2,)
    elif args.dataset == 'mnist':
        Xtrain, ytrain = mnist(args.n, train=True)
        Xtest, ytest = mnist(args.n, train=False)
        input_shape = (784,)
    else:
        raise ValueError('unknown dataset')
    
    # Construct models
    model_class = get_model(args.model + '_' + args.dataset)
    model = model_class(args.lr)
    
    # Construct trainer
    T = Trainer(input_shape, model)
    
    # Fit to data
    T.fit(Xtrain=Xtrain, n_epochs=args.n_epochs, batch_size=args.batch_size,
          warmup=args.warmup, beta=args.beta, iw_samples=args.iw_samples,
          logdir=logdir, ytrain=ytrain, Xtest=Xtest, ytest=ytest, 
          log_epoch=args.log_epoch, switch_epoch=args.switch_epoch) 