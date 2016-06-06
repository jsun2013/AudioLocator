# -*- coding: utf-8 -*-
"""
Created on Thu May 26 16:08:06 2016

@author: ReidW
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:26:24 2016

@author: ReidW
"""
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

import samples
reload(samples)
import spectral
reload(spectral)
import audiolearning
reload(audiolearning)
import mytimer as mt
reload(mt)

#PARAMETERS
FFT_BINS = 60;
FWIN = 25;
TWIN = 1;
NPERSEG = 1024;
TSUB = 1; #NOTE: this will be used only for training. Will experiment with test format later.
NSUB = 60; #This is just used for feature generation, doesn't really matter.
TEST_NSUB = 10;
#Last day of 'training' data
MONTH = 5;
DAY = 29;


#OTHER OPTIONS
METHOD = 'rbf' #'rbf' or 'logistic' or 'linear'
#TODO: ^not implemented!!
GET_CURVE=False
USE_PROB = False
ENSEMBLE=True
iters = 20;

#Save off training data through 05/16
train_file = 'trainingPhi_%02i%02i_%i_%i_%i_%i_%i_%i.pkl'%(MONTH,DAY,TSUB,NSUB,FFT_BINS,FWIN,TWIN,NPERSEG);


class phi1:
    LEN = 0;
    def __init__(self):
        #self.LEN = fft_bins;
        #self.LEN = 13;
        self.LEN = FFT_BINS+13;

    def get_phi(self,sample):
        '''
        Takes in a super_sample and returns a feature array. Breaks the super_sample
        down into samples. Each row of the returned value corresponds to a sample
        in the super sample.
        '''
        XSPED = spectral.getSupersampleSPED(sample,FFT_BINS,fwin=FWIN,twin = TWIN,nperseg = NPERSEG,spacing="log")
        XMFCC = spectral.getSampleMFCC(sample)
        return np.hstack((XMFCC,XSPED))
        #return XMFCC
        #return XSPED

def reshape_into_subsamples(X_all, Y_all, tsub, nsub):
    #SUGGESTED USAGE:
    #For test data, run feature extraction for max. number of subsamples per audio clip
    m_all, nsub_all, nfeat = np.shape(X_all);

    mult = int(np.floor(nsub_all/nsub)); #How many can we fit in?
    m = m_all*mult;
    X_new = np.zeros((m,nsub,nfeat));
    Y_new = np.repeat(Y_all,mult);
    #Fill in new matrix
    for i_new in range(m):
        i_all = int(np.floor(np.float(i_new)/mult));
        j_all = i_new - i_all*mult;
        X_new[i_new,:,:] = X_all[i_all,j_all*nsub:(j_all+1)*nsub,:];
        #for j in range(mult):
    return (X_new,Y_new);

if __name__ == "__main__":
    mt.tic()

    myPhi = phi1();
    #TODO: add other classifiers
    extractor = audiolearning.Classifier(myPhi)

    #First check if we have data. If not, generate
    if os.path.exists(train_file):
        with open(train_file,"rb") as myPkl:
            thisData = pickle.load(myPkl);
        X_train = thisData.X;
        Y_train = thisData.Y;
    else:
        all_samples = samples.getAllSamples(Tsub=TSUB,Nsub=NSUB,READ_IN=False) #
        train_samples = np.empty((0));
        #Choose only data up to this date
        for isample in all_samples:
            if isample.date.month < MONTH or (isample.date.month == MONTH and isample.date.day <= DAY):
                train_samples = np.append(train_samples,np.array([isample]))
        (X_train,Y_train) = extractor.extract_features(train_samples);
        thisData = samples.DataPhi();
        thisData.X=X_train;
        thisData.Y=Y_train;
        with open(train_file,"wb") as myPkl:
            pickle.dump(thisData,myPkl);
    #TRAIN CLASSIFIER
    #TODO: add other methods
    if ENSEMBLE:
        extractor.trainEnsemble1(train_samples=None, X_train=X_train, Y_train=Y_train,kernel='rbf',
                                    C=50000,gamma=1/(10000*float(myPhi.LEN)),probability=USE_PROB);
    else:
        extractor.trainSVMBatch(train_samples=None,X_train=X_train, Y_train=Y_train,
                                kernel='rbf',C=5000,gamma=1/(10000*float(myPhi.LEN)),probability=USE_PROB);
    #NOTE: using short NSUB for training does change the training error evaluation


    #TEST PARAMETER: how many subsamples do we use?
    TEST_NSUB_ALL = int(np.ceil(60/TSUB)); #Fit as many subsamples in per audio clip as possible
    test_file = 'testPhi_%02i%02i_%i_%i_%i_%i_%i_%i.pkl'%(MONTH,DAY,TSUB,TEST_NSUB_ALL,FFT_BINS,FWIN,TWIN,NPERSEG);
    if os.path.exists(test_file):
        with open(test_file,"rb") as myPkl:
            thisData = pickle.load(myPkl);
        X_test = thisData.X;
        Y_test = thisData.Y;
    else:
        all_samples = samples.getAllSamples(Tsub=TSUB,Nsub=TEST_NSUB_ALL,READ_IN=False) #
        test_samples = np.empty((0));
        for isample in all_samples:
            if isample.date.month > MONTH or (isample.date.month == MONTH and isample.date.day > DAY):
                test_samples = np.append(test_samples,np.array([isample]))
        (X_test,Y_test) = extractor.extract_features(test_samples);
        thisData = samples.DataPhi();
        thisData.X=X_test;
        thisData.Y=Y_test;
        with open(test_file,"wb") as myPkl:
            pickle.dump(thisData,myPkl);
    if GET_CURVE==False:
        (X_test,Y_test) = reshape_into_subsamples(X_test,Y_test,TSUB,TEST_NSUB);
        totalErr = 0;
        for j in range(iters):
            if ENSEMBLE:
                totalErr += extractor.testClassifierEnsemble(test_samples=None,X_test=X_test,Y_test=Y_test);
            else:
                totalErr += extractor.testClassifier(test_samples=None,X_test=X_test,Y_test=Y_test,probability=USE_PROB);
        totalErr =totalErr/iters;
        print("Average Generalized Test Error = %0.04f"%totalErr)
    else:
        test_nsub_arr = np.array([1, 2, 4, 5, 10, 12, 15, 20, 25, 30,45,60],dtype=np.int8);
        test_nsub_errs = np.zeros(np.size(test_nsub_arr));
        for (i_nsub, test_nsub) in enumerate(test_nsub_arr):
            (jX_test,jY_test) = reshape_into_subsamples(X_test,Y_test,TSUB,test_nsub);
            totalErr = 0;
            for j in range(iters):
                if ENSEMBLE:
                    totalErr += extractor.testClassifierEnsemble(test_samples=None,X_test=X_test,Y_test=Y_test);
                else:
                    totalErr += extractor.testClassifier(test_samples=None,X_test=jX_test,Y_test=jY_test,probability=USE_PROB);
            test_nsub_errs[i_nsub] = totalErr/iters;
        fig = plt.figure();
        plt.scatter(test_nsub_arr,test_nsub_errs);
        plt.xlabel('# Subsamples');
        plt.ylabel('Test Error on New Data');
    mt.toc()
