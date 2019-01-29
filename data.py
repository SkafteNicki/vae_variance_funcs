#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:52:39 2019

@author: nsde
"""

#%%
import torch
import numpy as np

#%%
def two_moons(N=1000):
    angle = 2*np.pi*np.random.uniform(0, 1/2, N)
    x = np.cos(angle) + 0.1*np.random.randn(N)
    y = np.sin(angle) + 0.1*np.random.randn(N)
    angle = 2*np.pi*np.random.uniform(1/2, 1, N)
    x2 = np.cos(angle) + 0.1*np.random.randn(N) + 0.7
    y2 = np.sin(angle) + 0.1*np.random.randn(N)
    X = np.stack((np.hstack([x,x2]),np.hstack([y,y2]))).T.astype(np.float32)
    y = np.concatenate((np.zeros(N,), np.ones(N,)))
    return torch.tensor(X).to(torch.float32), torch.tensor(y).to(torch.float32)