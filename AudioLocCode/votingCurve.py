# -*- coding: utf-8 -*-
"""
Created on Thu Jun 02 17:52:56 2016

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

feature_file = 'alldaysPhi_%i_%i_%i_%i_%i_%i.pkl'%(TSUB,NSUB,FFT_BINS,FWIN,TWIN,NPERSEG);

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
    ens_svm = audiolearning.Classifier(myPhi);

    #First check if we have data. If not, generate
    if os.path.exists(feature_file):
        with open(feature_file,"rb") as myPkl:
            dated_phi = pickle.load(myPkl);
        n_days = len(dated_phi);
    else:
        all_samples = samples.getAllSamples(Tsub=TSUB,Nsub=NSUB,READ_IN=False) #
        dated_samples = np.empty((0));
        #Choose only data up to this date
        #Split according to day
        dates = [];
        for isample in all_samples:
            if isample.date in dates:
                dated_samples[dates.index(isample.date)] = np.append(dated_samples[dates.index(isample.date)],isample);
            else:
                dated_samples = np.append(dated_samples, np.array([isample]));
                dates.append(isample.date);
        n_days = len(dates);
        dated_phi = []
        for iday in range(n_days):
            (X_train,Y_train) = ens_svm.extract_features(dated_samples[iday]);

            thisData = samples.DataPhi();
            thisData.X=X_train;
            thisData.Y=Y_train;
            dated_phi.append(thisData);
        with open(feature_file,"wb") as myPkl:
            pickle.dump(dated_phi,myPkl);

    test_nsub_arr = np.array([1, 2, 4, 5, 10, 12, 15, 20, 25, 30,45,60],dtype=np.int8);
    test_nsub_errs = np.zeros(np.size(test_nsub_arr));
    test_total_samples = np.zeros(np.size(test_nsub_arr));

    for iday in range(n_days):
        #Training data is all but this day
        _, nsub, nfeat = np.shape(dated_phi[0].X);
        X_train = np.empty((0,nsub, nfeat));
        Y_train = np.empty((0));
        for jday in range(n_days):
            if jday != iday:
                X_train = np.concatenate((X_train, dated_phi[jday].X),axis=0);
                Y_train = np.append(Y_train, dated_phi[jday].Y);

        ens_svm.trainEnsemble1(train_samples=None, X_train=X_train, Y_train=Y_train,kernel='rbf',
                                        C=50000,gamma=1/(10000*float(myPhi.LEN)),probability=False);

        X_test = dated_phi[iday].X; Y_test = dated_phi[iday].Y;
        for (i_nsub, test_nsub) in enumerate(test_nsub_arr):
            (iX_test,iY_test) = reshape_into_subsamples(X_test,Y_test,TSUB,test_nsub);
            jnsamp = np.size(Y_test);
            ens_svm_err = ens_svm.testClassifierEnsemble(test_samples=None,X_test=iX_test,Y_test=iY_test);
            test_nsub_errs[i_nsub] += jnsamp*ens_svm_err;
            test_total_samples[i_nsub] += jnsamp;

    test_nsub_errs = np.divide(test_nsub_errs,test_total_samples);

    fig = plt.figure();
    plt.plot(test_nsub_arr,test_nsub_errs,marker="p",markersize=16,linewidth=4);
    plt.xlabel('# Subsamples',fontsize=20);
    plt.ylabel('Generalized Test Error',fontsize=20);
    plt.ylim((0.1,0.3))
    mt.toc()
