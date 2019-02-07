#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:23:55 2019

@author: nsde
"""

#%%
import sys
sys.path.append("..")

#%%
def get_model(model_name):
    import os
    import importlib
    files = os.listdir(os.path.dirname(os.path.realpath(__file__)))
    py_files = [f for f in files if '.py' in f and '__init__.py' not in f]
    
    models = dict()
    for file in py_files:
        i = importlib.import_module('.' + file[:-3], package='models')
        modules = dir(i)
        for m in modules:
            if 'VAE' in m and '_base' not in m:
                models[m.lower()] = getattr(i, m)
                
    assert model_name in models, 'Model "' + model_name + '" not found, choose between: ' \
        + ', '.join([k for k in models.keys()])

    return models[model_name]

#%%
if __name__ == '__main__':
    pass