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

FFT_BINS = 40;

class phi1:
    LEN = 0;
    def __init__(self,fft_bins):
        self.LEN = fft_bins+1;
        self.fft_bins = fft_bins;

    def get_phi(self,sample):
        '''
        Takes in a super_sample and returns a feature array. Breaks the super_sample
        down into samples. Each row of the returned value corresponds to a sample
        in the super sample.
        '''

        XSPED = spectral.getSupersampleSPED(sample,self.fft_bins,fwin=25,twin = 1,spacing="log")
        XMean = np.zeros((np.shape(XSPED)[0],1))
        subsamples = sample.getSubsamples();
        for j,data in enumerate(subsamples):
            #XFFT[j,:] = F_all[j,:]
            XMean[j] = np.mean(data)
        return np.hstack((XSPED,XMean))
if __name__ == "__main__":
    mt.tic()
    all_samples = samples.getAllSamples(Tsub=2,Nsub=5,key="phone",val="Reid",READ_IN=True) #2 second subsamples, 5 per sample

    np.random.shuffle(all_samples);
    numTrain = int(round(2*len(all_samples)/3))
    train_samples = all_samples[:numTrain]
    test_samples = all_samples[numTrain:]

    nfft_bins = FFT_BINS;
    myPhi = phi1(nfft_bins);
    logistic_classifier = audiolearning.Classifier(myPhi);
    logistic_classifier.trainLogitBatch2(train_samples,test_samples);

    n_test = len(test_samples);
    test_actual = np.zeros((n_test,1))
    test_hat = np.zeros((n_test,1))
    for (i,isup) in enumerate(test_samples):
        test_actual[i] = isup.region
        test_hat[i] = logistic_classifier.make_prediction(isup)

    print("-----------------------------------------------------")
    print("-------------------Testing Error:-------------------")
    for region in range(7):
        actual = test_actual[test_actual == region]
        pred = test_hat[test_actual == region]
        if len(actual)==0:
            print("  ->No test samples from Region %d"%region)
            continue
        err = 1 - float(sum(actual == pred))/len(actual)
        print "Error for region %d: %.4f" % (region,err)
    totalErr = 1 - float(sum(test_actual == test_hat))/n_test
    print "---- Total Testing Error: %.4f" % totalErr

    mt.toc()
