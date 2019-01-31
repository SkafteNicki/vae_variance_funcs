#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:23:55 2019

@author: nsde
"""

#%%
from .vae_full import VAE_full
from .vae_diag import VAE_diag
from .vae_single import VAE_single
from .vae_student import VAE_student
from .vae_rbf import VAE_rbf
from .vae_experimental import VAE_experimental

#%%
def get_model(model_name):
    models = {'vae_full': VAE_full,
              'vae_diag': VAE_diag,
              'vae_single': VAE_single,
              'vae_student': VAE_student,
              'vae_rbf': VAE_rbf,
              'vae_experimental': VAE_experimental
              }
    assert (model_name in models), 'Model not found, choose between: ' \
            + ', '.join([k for k in models.keys()])
    return models[model_name]

#%%
if __name__ == '__main__':
    pass