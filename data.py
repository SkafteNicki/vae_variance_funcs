#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:52:39 2019

@author: nsde
"""

#%%
import torch
import numpy as np
import gzip
import os
from urllib.request import urlretrieve

#%%
def two_moons(N=1000, train=True):
    angle = 2*np.pi*np.random.uniform(0, 1/2, N)
    x = np.cos(angle) + 0.1*np.random.randn(N)
    y = np.sin(angle) + 0.1*np.random.randn(N)
    angle = 2*np.pi*np.random.uniform(1/2, 1, N)
    x2 = np.cos(angle) + 0.1*np.random.randn(N) + 0.7
    y2 = np.sin(angle) + 0.1*np.random.randn(N)
    X = np.stack((np.hstack([x,x2]),np.hstack([y,y2]))).T.astype(np.float32)
    y = np.concatenate((np.zeros(N,), np.ones(N,)))
    return torch.tensor(X).to(torch.float32), torch.tensor(y).to(torch.float32)

#%%
def mnist(N = 10000, train=True, path=None):
    url = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

    if path is None:
        path = os.getcwd() + '/mnist_data'

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download any missing files
    for file in files:
        if file not in os.listdir(path):
            urlretrieve(url + file, os.path.join(path, file))
            print("Downloaded %s to %s" % (file, path))

    def _images(path):
        """Return images loaded locally."""
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 784).astype('float32') / 255

    def _labels(path):
        """Return labels loaded locally."""
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)

        return integer_labels

    train_images = _images(os.path.join(path, files[0]))
    train_labels = _labels(os.path.join(path, files[1]))
    test_images = _images(os.path.join(path, files[2]))
    test_labels = _labels(os.path.join(path, files[3]))
    
    if train:
        data = torch.tensor(train_images)
        targets = torch.tensor(train_labels)
    else:
        data = torch.tensor(test_images)
        targets = torch.tensor(test_labels)
    
    # Extract N points pr class
    newdata, newtargets = [ ], [ ]
    counter = 10 * [0]
    for x, y in zip(data, targets):
        if counter[y] < N:
            newdata.append(x)
            newtargets.append(y)
            counter[y] += 1
    data = torch.stack(newdata, dim=0)
    targets = torch.stack(newtargets, dim=0)
    
    return data, targets