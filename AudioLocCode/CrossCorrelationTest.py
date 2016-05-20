# -*- coding: utf-8 -*-
"""
Created on Fri May 20 12:06:50 2016

@author: James
"""

import samples
import numpy as np

train_samples = samples.getAllSamples()
X = np.array([])
Y = np.array([])
'''Build autocorrelation as features'''
for sample,i in zip(train_samples,range(len(train_samples))):
    Y[i] = sample.region
    
    X = np.vstack((X,))