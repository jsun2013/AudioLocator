# -*- coding: utf-8 -*-
"""
Created on Fri May 27 01:28:40 2016

@author: ReidW
"""
import os
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
LOAD_DATA = True;
data_file = 'spedPhi_2_10_60_45_1_1024.pkl'
frac_test = 0.2;

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


    #Create a classifier instance with your feature extraction method
    myPhi = phi1(FFT_BINS);
    extractor = audiolearning.Classifier(myPhi);

    #Extract all features once
    if LOAD_DATA and os.path.exists(data_file):
        with open(data_file,"rb") as myPkl:
            thisData = pickle.load(myPkl);
        X = thisData.X;
        Y = thisData.Y;
        nsamp, _, _ = np.shape(X);
    else:
        #Collect all samples, figure out how many are reserved for test in X-validation
        all_samples = samples.getAllSamples(Tsub=2,Nsub=10,READ_IN=False) #
        nsamp = len(all_samples);

        (X,Y) = extractor.extract_features(all_samples)
        thisData = samples.DataPhi();
        thisData.X=X;
        thisData.Y=Y;
        with open(data_file,"wb") as myPkl:
            pickle.dump(thisData,myPkl);

    num_test = int(round(frac_test*nsamp));

    #Shuffle up the order
    inds = range(nsamp);
    np.random.shuffle(inds);

    X_train = X[inds[num_test:],:,:];
    Y_train = Y[inds[num_test:]];

    X_test = X[inds[:num_test],:,:];
    Y_test = Y[inds[:num_test]];

    extractor.trainSVMBatch(train_samples=None, X_train=X_train, Y_train=Y_train,kernel='rbf',
                                C=50000,gamma=1/(10000*float(myPhi.LEN)) );

    extractor.testClassifier(test_samples=None,X_test=X_test,Y_test=Y_test);
