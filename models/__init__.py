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
from .vae_cluster import VAE_cluster

#%%
def get_model(model_name):
    models = {'vae': VAE_full,
              'vitae_ci': VAE_diag,
              'vitae_ui': VAE_single,
              'vae_student': VAE_student,
              'vae_cluster': VAE_cluster
              }
    assert (model_name in models), 'Model not found, choose between: ' \
            + ', '.join([k for k in models.keys()])
    return models[model_name]

#%%
if __name__ == '__main__':
    pass