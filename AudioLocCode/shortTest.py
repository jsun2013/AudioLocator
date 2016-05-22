# -*- coding: utf-8 -*-
"""
Created on Sat May 21 18:24:01 2016

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

np.random.seed(25)

class phi0:
    LEN = 0;
    def __init__(self,fft_bins):
        self.LEN = fft_bins+1;
        self.fft_bins = fft_bins;

    def get_phi(self,super_sample, audio_dur=60, sample_length=10):
        '''
        Takes in a super_sample and returns a feature array. Breaks the super_sample
        down into samples. Each row of the returned value corresponds to a sample
        in the super sample.
        '''
        samples_per = super_sample.N;

        XFFT = np.zeros((2*samples_per,self.fft_bins))
        XMean = np.zeros((2*samples_per,1))

        _,XFFT = spectral.getSupersampleFFT(super_sample,self.fft_bins,spacing="log")
        for j,data in enumerate(super_sample.samples):
            #XFFT[j,:] = F_all[j,:]
            XMean[j] = np.mean(data)
        return np.hstack((XFFT[:j+1,:],XMean[:j+1,:]))

sample_length = 2 # Seconds
all_samples = samples.getAllSamples(T=sample_length,N=15,key="phone",val="Reid")

np.random.shuffle(all_samples);
numTrain = int(round(2*len(all_samples)/3))
train_samples = all_samples[:numTrain]
test_samples = all_samples[numTrain:]

nfft_bins = 80;
myPhi = phi0(nfft_bins);
logistic_classifier = audiolearning.Classifier(myPhi);
logistic_classifier.trainLogitBatch2(train_samples,test_samples);