# -*- coding: utf-8 -*-
"""
Created on Thu May 26 21:28:43 2016

@author: ReidW
"""
import os
import numpy as np
from sklearn import linear_model
from statsmodels.tsa import stattools
from scipy import stats
import matplotlib.pyplot as plt
from scikits.talkbox import features
import pickle

import samples
reload(samples)
import spectral
reload(spectral)
import audiolearning
reload(audiolearning)
import mytimer as mt

LOAD_DATA = True;
data_file = 'spedPhi_2_10_180_45_1_1024.pkl'
frac_test = 0.2;

class spedPhi:
    LEN = 0;
    def __init__(self,fft_bins):
        self.LEN = fft_bins;
        self.fft_bins = fft_bins;

    def get_phi(self,sample):
        '''
        Takes in a super_sample and returns a feature array. Breaks the super_sample
        down into samples. Each row of the returned value corresponds to a sample
        in the super sample.
        '''

        XSPED = spectral.getSupersampleSPED(sample,self.fft_bins,fwin=25,twin = 1,nperseg = 1024,spacing="log")
        return XSPED


mt.tic();

nfft_bins_start = 180; #Full features
nfft_bins_end = 40; #Reduced

#bin_inds = np.empty((nfft_bins_end,1));
bin_inds = np.empty(0,dtype=np.int8);
feat_errs = np.empty(nfft_bins_end);

myPhi = spedPhi(nfft_bins_start);
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

numTest = int(round(frac_test*nsamp));


inds = range(nsamp);

prog = 0;
tot = nfft_bins_end*nfft_bins_start;
nupdate = int(tot/23);

for nfeat in range(nfft_bins_end):
    #Separate training and test data
    np.random.shuffle(inds);
    X_train = X[inds[numTest:],:,:];
    Y_train = Y[inds[numTest:]];

    X_test = X[inds[:numTest],:,:];
    Y_test = Y[inds[:numTest]];

    ifeat_errs = np.zeros(nfft_bins_start);
    #Retrain our classifier
    for ifeat in range(nfft_bins_start):
        prog+=1;
        if prog%nupdate==0:
            print("%d%%...")
        if (bin_inds==ifeat).any():
            #We have already chosen this feature. Keep moving along
            #Set error to 100% so it isn't chosen as best feature
            ifeat_errs[ifeat]=1;
            continue;
        indsJ = np.append(bin_inds,[ifeat]); #add this feature to existing
        indsJ = np.int(indsJ);
        #Use reduced feature set
        pX_train = X_train[:,:,indsJ];
        pX_test = X_test[:,:,indsJ];
        if np.size(bin_inds)==0:
            pX_train = np.expand_dims(pX_train,2)
            pX_test = np.expand_dims(pX_test,2)

        extractor.trainSVMBatch(train_samples=None, X_train=pX_train, Y_train=Y_train,kernel='rbf',
                                C=50000,gamma=1/(10000*float(np.size(bin_inds)+1)) );

        ifeat_errs[ifeat] = extractor.testClassifier(test_samples=None,X_test=pX_test,Y_test=Y_test);

    #Now find additional feature that did best on test error
    ifeat_min = np.argmin(ifeat_errs);
    bin_inds = np.append(bin_inds,ifeat_min);
    feat_errs[nfeat] = ifeat_errs[ifeat_min];


mt.toc();

