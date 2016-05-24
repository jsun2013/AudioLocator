# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:26:24 2016

@author: ReidW
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

FFT_BINS = 60;

class phi1:
    LEN = 0;
    def __init__(self,fft_bins):
        #self.LEN = fft_bins;
        self.LEN = fft_bins+13;
        self.fft_bins = fft_bins;

    def get_phi(self,sample):
        '''
        Takes in a super_sample and returns a feature array. Breaks the super_sample
        down into samples. Each row of the returned value corresponds to a sample
        in the super sample.
        '''

        XSPED = spectral.getSupersampleSPED(sample,self.fft_bins,fwin=25,twin = 1,nperseg = 1024,spacing="log")
        XMFCC = spectral.getSampleMFCC(sample)
        return np.hstack((XMFCC,XSPED))
        #return XSPED

if __name__ == "__main__":
    mt.tic()
    all_samples = samples.getAllSamples(Tsub=2,Nsub=10,key="phone",val="Reid",READ_IN=False) #2 second subsamples, 5 per sample

    np.random.shuffle(all_samples);
    numTrain = int(round(2*len(all_samples)/3))
    train_samples = all_samples[:numTrain]
    test_samples = all_samples[numTrain:]

    nfft_bins = FFT_BINS;
    myPhi = phi1(nfft_bins);
    """
    logistic_classifier = audiolearning.Classifier(myPhi);
    logistic_classifier.trainLogitBatch(train_samples);
    """
    svm_classifier = audiolearning.Classifier(myPhi)
    svm_classifier.trainSVMBatch(train_samples,kernel='linear',C=1000)
    #svm_classifier.trainSVMBatch(train_samples,kernel='rbf',C=500,gamma=10/float(myPhi.LEN))
    svm_classifier.testClassifier(test_samples)


    mt.toc()
