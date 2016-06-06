# -*- coding: utf-8 -*-
"""
Created on Mon May 30 19:19:03 2016

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
import general_hme as hme
reload(hme)

#PARAMETERS
FFT_BINS = 60;
FWIN = 25;
TWIN = 1;
NPERSEG = 1024;
TSUB = 1;
NSUB = 10;

frac_test = 0.2;
ITERS=10; #Due to randomness when tie-breaking, average over some runs

data_file = 'spedPhi_%i_%i_%i_%i_%i_%i.pkl'%(TSUB,NSUB,FFT_BINS,FWIN,TWIN,NPERSEG);

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
    rbf_svm = audiolearning.Classifier(myPhi);
    lin_svm = audiolearning.Classifier(myPhi);
    lin_log = audiolearning.Classifier(myPhi);
    ens_svm = audiolearning.Classifier(myPhi);
    rf_clf = audiolearning.Classifier(myPhi);

    if os.path.exists(data_file):
        with open(data_file,"rb") as myPkl:
            thisData = pickle.load(myPkl);
        X = thisData.X;
        Y = thisData.Y;
        nsamp, _, _ = np.shape(X);
    else:
        #Collect all samples, figure out how many are reserved for test in X-validation
        all_samples = samples.getAllSamples(Tsub=TSUB,Nsub=NSUB,READ_IN=False) #
        nsamp = len(all_samples);

        (X,Y) = rbf_svm.extract_features(all_samples)
        thisData = samples.DataPhi();
        thisData.X=X;
        thisData.Y=Y;
        with open(data_file,"wb") as myPkl:
            pickle.dump(thisData,myPkl);

    num_test = int(round(frac_test*nsamp));

    totalErr = np.zeros(5);
    for i_iter in range(ITERS):
        #Shuffle up the order
        inds = range(nsamp);
        np.random.shuffle(inds);

        X_train = X[inds[num_test:],:,:];
        Y_train = Y[inds[num_test:]];

        X_test = X[inds[:num_test],:,:];
        Y_test = Y[inds[:num_test]];

        rbf_svm.trainSVMBatch(train_samples=None,X_train=X_train, Y_train=Y_train,
                                    kernel='rbf',C=5000,gamma=1/(10000*float(myPhi.LEN)));
        lin_svm.trainSVMBatch(train_samples=None,X_train=X_train, Y_train=Y_train,
                                    kernel='linear',C=5000);
        lin_log.trainLogitBatch(train_samples=None,X_train=X_train, Y_train=Y_train);
        ens_svm.trainEnsemble1(train_samples=None, X_train=X_train, Y_train=Y_train,kernel='rbf',
                                    C=5000,gamma=1/(10000*float(myPhi.LEN)),probability=False);
        rf_clf.trainRandomForestBatch(train_samples=None, X_train=X_train, Y_train=Y_train,n_estimators=100,max_features=20,n_jobs=-1);

        totalErr[0] += rbf_svm.testClassifier(test_samples=None,X_test=X_test,
                                        Y_test=Y_test);
        totalErr[1] += lin_svm.testClassifier(test_samples=None,X_test=X_test,
                                        Y_test=Y_test);
        totalErr[2] += lin_log.testClassifier(test_samples=None,X_test=X_test,
                                        Y_test=Y_test);
        totalErr[3] += ens_svm.testClassifierEnsemble(test_samples=None,X_test=X_test,
                                        Y_test=Y_test);
        totalErr[4] += rf_clf.testClassifier(test_samples=None,X_test=X_test,
                                        Y_test=Y_test);

    totalErr = np.divide(totalErr,ITERS);
    print("Final Average Generalization Errors:")

    np.set_printoptions(precision=5);
    print totalErr;
    mt.toc()