# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:26:24 2016

@author: ReidW
"""

import numpy as np
from sklearn import linear_model
from statsmodels.tsa import stattools
from scipy import stats
import pickle

import samples
reload(samples)
import spectral
reload(spectral)
import audiolearning
reload(audiolearning)
import mytimer as mt

FFT_BINS = 60;
LOAD_DATA = False;
data_file = 'spedPhi_2_10_60_45_1_1024.pkl'

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
    if LOAD_DATA==False:
        #all_samples = samples.getAllSamples(Tsub=2,Nsub=10,key="phone",val="Reid",READ_IN=True) # 0% test error??
        all_samples = samples.getAllSamples(Tsub=2,Nsub=10,READ_IN=False) #
        #all_samples = samples.getAllSamples(Tsub=2,Nsub=5,key="phone",val="James",READ_IN=True)
        #all_samples = samples.getAllSamples(Tsub=2,Nsub=10,READ_IN=False)
        #all_samples = samples.getAllSamples(Tsub=2,Nsub=5,key="phone",val="Reid",READ_IN=False) #

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
        #svm_classifier.trainSVMBatch(train_samples,kernel='linear',C=1000)
        #svm_classifier.trainSVMBatch(train_samples,kernel='rbf',C=500,gamma=1/(10000*float(myPhi.LEN))); #Set 3: 15% error
        svm_classifier.trainSVMBatch(train_samples,kernel='rbf',C=50000,gamma=1/(10000*float(myPhi.LEN)));
        #svm_classifier.trainSVMBatch(train_samples,kernel='rbf',C=500,gamma=1/(1000*float(myPhi.LEN))); #0.07 error
        #svm_classifier.trainSVMBatch(train_samples,kernel='rbf',C=500,gamma=1/(100*float(myPhi.LEN))); #0.08 error
        svm_classifier.testClassifier(test_samples)

    else:
        with open(data_file,'rb') as myPkl:
            myPhi = pickle.load(myPkl)

        nSamples = np.shape(myPhi.X)[0];
        inds = range(nSamples);
        np.random.shuffle(inds);

        numTrain = int(round(2*nSamples/3));
        test_inds = inds[numTrain:];
        train_inds = inds[:numTrain];

        X_test = myPhi.X[test_inds,:];
        Y_test = myPhi.Y[test_inds,:];

        X_train = myPhi.X[train_inds,:];
        Y_train = myPhi.Y[train_inds,:];


        svm_classifier = audiolearning.Classifier(myPhi)
        svm_classifier.trainSVMBatch(test_samples=None, X_train=X_train, Y_train=Y_train,kernel='rbf',
                                C=50000,gamma=1/(10000*float(myPhi.LEN)) );

        svm_classifier.testClassifier(test_samples=None,X_test=X_test,Y_test=Y_test);

    mt.toc()
