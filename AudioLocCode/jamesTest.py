# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:26:24 2016

@author: james
"""

import numpy as np
from sklearn import linear_model
from statsmodels.tsa import stattools
from scipy import stats

import samples
reload(samples)
import spectral
reload(spectral)
import audiolearning
reload(audiolearning)
import mytimer as mt
from scikits.talkbox import features

FFT_BINS = 40;
ACF_LAGS = 40

class phi1:
    LEN = 0;
    def __init__(self,acf_lags,nceps):
#        self.LEN = nceps+acf_lags+1
        self.LEN = nceps
        self.acf_lags = acf_lags

    def get_phi(self,sample):
        '''
        Takes in a super_sample and returns a feature array. Breaks the super_sample
        down into samples. Each row of the returned value corresponds to a sample
        in the super sample.
        '''

#        XSPED = spectral.getSupersampleSPED(super_sample,self.fft_bins,spacing="log")
#        XAcf = spectral.getSampleACF(sample,self.acf_lags)
#        XMean = np.zeros((sample.Nsub,1))
        XMFCC = spectral.getSampleMFCC(sample)
#        XAcf = np.zeros((sample.Nsub,self.acf_lags+1))
#        XMFCC = np.zeros((sample.Nsub,13))
#        subsamples = sample.getSubsamples()
#        for j,data in enumerate(subsamples):      
#            XMFCC[j,:] = spectral.getSignalMFCC(data)
#            XAcf[j,:] = spectral.getSignalACF(data,self.acf_lags)
#        return np.hstack((XMFCC,XMean,XAcf))
#        return np.hstack((XMFCC,XAcf))
        return XMFCC
mt.tic()
all_samples = samples.getAllSamples(Tsub=2,Nsub=5,key="phone",val="Reid",READ_IN=True) #2 second subsamples, 5 per sample
#all_samples = samples.getAllSamples(T=5,N=10) #2 second samples, 20 samples per supersample
#  all_samples = samples.getAllSamples(T=2,N=25,key="phone",val="James") #2 second samples, 20 samples per supersample

np.random.shuffle(all_samples);
numTrain = int(round(2*len(all_samples)/3))
train_samples = all_samples[:numTrain]
test_samples = all_samples[numTrain:]

nfft_bins = FFT_BINS;
myPhi = phi1(ACF_LAGS,13);
logistic_classifier = audiolearning.Classifier(myPhi);
logistic_classifier.trainLogitBatch(train_samples,C=300);
logistic_classifier.testClassifier(test_samples)

svm_classifier = audiolearning.Classifier(myPhi)
svm_classifier.trainSVMBatch(train_samples,test_samples,C=500)
svm_classifier.testClassifier(test_samples)



mt.toc()
