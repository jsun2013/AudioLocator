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
    def __init__(self,fft_bins,acf_lags):
        self.LEN = 13
        self.fft_bins = fft_bins
        self.acf_lags = acf_lags

    def get_phi(self,super_sample):
        '''
        Takes in a super_sample and returns a feature array. Breaks the super_sample
        down into samples. Each row of the returned value corresponds to a sample
        in the super sample.
        '''

#        XSPED = spectral.getSupersampleSPED(super_sample,self.fft_bins,spacing="log")
        XAcf = np.zeros((super_sample.N,ACF_LAGS+1))
        XMean = np.zeros((super_sample.N,1))
        XMFCC = np.zeros((super_sample.N,13))
        samples = super_sample.readoutSamples();
        for j,data in enumerate(samples):
            #XFFT[j,:] = F_all[j,:]
#            XAcf[j,:] = stattools.acf(data,nlags=self.acf_lags,fft=True)
#            XMean[j] = np.mean(data)
            temp = features.mfcc(data,fs=44100)[0]
            temp[~np.isfinite(temp)] = 0            
            XMFCC[j,:] = np.mean(temp,0)
#        return np.hstack((XMFCC,XMean,XAcf))
        return XMFCC

mt.tic()
all_samples = samples.getAllSamples(T=2,N=25) #2 second samples, 20 samples per supersample
#all_samples = samples.getAllSamples(T=5,N=10) #2 second samples, 20 samples per supersample
#  all_samples = samples.getAllSamples(T=2,N=25,key="phone",val="James") #2 second samples, 20 samples per supersample

np.random.shuffle(all_samples);
numTrain = int(round(2*len(all_samples)/3))
train_samples = all_samples[:numTrain]
test_samples = all_samples[numTrain:]

nfft_bins = FFT_BINS;
myPhi = phi1(nfft_bins,ACF_LAGS);
logistic_classifier = audiolearning.Classifier(myPhi);
logistic_classifier.trainLogitBatch(train_samples,C=300);
logistic_classifier.testClassifier(test_samples)

svm_classifier = audiolearning.Classifier(myPhi)
svm_classifier.trainSVMBatch(train_samples,test_samples,C=500)
svm_classifier.testClassifier(test_samples)



#n_test = len(test_samples);
#test_actual = np.zeros((n_test,1))
#test_hat = np.zeros((n_test,1))
#for (i,isup) in enumerate(test_samples):
#    test_actual[i] = isup.region
#    test_hat[i] = svm_classifier.make_prediction(isup)
#
#print("-----------------------------------------------------")
#print("-------------------Testing Error:-------------------")
#for region in range(7):
#    actual = test_actual[test_actual == region]
#    pred = test_hat[test_actual == region]
#    if len(actual)==0:
#        print("  ->No test samples from Region %d"%region)
#        continue
#    err = 1 - float(sum(actual == pred))/len(actual)
#    print "Error for region %d: %.4f" % (region,err)
#totalErr = 1 - float(sum(test_actual == test_hat))/n_test
#print "---- Total Testing Error: %.4f" % totalErr

mt.toc()
