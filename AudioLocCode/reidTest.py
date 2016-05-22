# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:26:24 2016

@author: ReidW
"""

import samples
reload(samples)
import numpy as np
from sklearn import linear_model
from statsmodels.tsa import stattools
import spectral
reload(spectral)
import audiolearning
reload(audiolearning)
from scipy import stats

SAMPLE_LEN = 2; #seconds
FFT_BINS = 50;


def phi(super_sample):
    '''
    Takes in a super_sample and returns a feature array. Breaks the super_sample
    down into samples. Each row of the returned value corresponds to a sample
    in the super sample.
    '''

    _, F_all = spectral.getSupersampleFFT(super_sample,FFT_BINS)



sups = samples.getAllSamples(T=SAMPLE_LEN,N=2,key="phone",val="Reid") #10 second samples, 2 samples per supersample
spectral.getSupersampleSPED(sups[0],50);