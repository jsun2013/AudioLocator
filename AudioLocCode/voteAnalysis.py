# -*- coding: utf-8 -*-
"""
Created on Fri May 27 01:28:40 2016

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
TSUB = 2;
NSUB = 5;

data_file = 'spedPhi_%i_%i_%i_%i_%i_%i.pkl'%(TSUB,NSUB,FFT_BINS,FWIN,TWIN,NPERSEG);
LOAD_DATA = True;

#Other Settings
frac_test = 0.2;
iters = 10;


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

        XSPED = spectral.getSupersampleSPED(sample,self.fft_bins,fwin=FWIN,twin = TWIN,nperseg = NPERSEG,spacing="log")
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
        all_samples = samples.getAllSamples(Tsub=TSUB,Nsub=NSUB,READ_IN=False) #
        nsamp = len(all_samples);

        (X,Y) = extractor.extract_features(all_samples)
        thisData = samples.DataPhi();
        thisData.X=X;
        thisData.Y=Y;
        with open(data_file,"wb") as myPkl:
            pickle.dump(thisData,myPkl);

    num_test = int(round(frac_test*nsamp));
    vote_rec_per = [[np.empty((0,2)), np.empty((0,2))] for i in range(7)]

    for jiter in range(iters):
        #Shuffle up the order
        inds = range(nsamp);
        np.random.shuffle(inds);

        X_train = X[inds[num_test:],:,:];
        Y_train = Y[inds[num_test:]];

        X_test = X[inds[:num_test],:,:];
        Y_test = Y[inds[:num_test]];

        extractor.trainSVMBatch(train_samples=None, X_train=X_train, Y_train=Y_train,kernel='rbf',
                                    C=50000,gamma=1/(10000*float(myPhi.LEN)) );

        (totalErr,test_actual,test_hat,vote_rec) = extractor.testClassifier(test_samples=None,X_test=X_test,
                                            Y_test=Y_test,get_conf_mat=False,return_rec=True);

        for (i,jlabel) in enumerate(test_actual):
            if test_hat[i]==jlabel:
                #Correct Prediction
                this_count = np.bincount(vote_rec[i]);
                hit0 = this_count[jlabel];
                this_count[jlabel]=0;
                hit1 = max(this_count); #Runner up vote.
                #vote_rec_per[jlabel][0].append([hit0, hit1]);
                vote_rec_per[jlabel][0] = np.append(vote_rec_per[jlabel][0],np.array([[hit0, hit1]]),axis=0);
            else:
                #Incorrect Prediction
                this_count = np.bincount(vote_rec[i]);
                #hit0 = this_count[jlabel]; #Actual label
                #hit1 = max(this_count); #Winning vote.
                hit0 = this_count[test_hat[i]]; #Winning vote
                this_count[test_hat[i]]=0;
                hit1 = max(this_count); #Runner up vote.
                #vote_rec_per[jlabel][1].append([hit0, hit1]);
                vote_rec_per[jlabel][1] = np.append(vote_rec_per[jlabel][1],np.array([[hit0, hit1]]),axis=0);
    total_del_correct = np.empty(0);
    total_del_incorrect = np.empty(0);
    for ireg in range(7):
        del_correct = vote_rec_per[ireg][0][:,0] - vote_rec_per[ireg][0][:,1];
        del_incorrect = vote_rec_per[ireg][1][:,0] - vote_rec_per[ireg][1][:,1];
        #Add to overall total
        total_del_correct = np.append(total_del_correct,del_correct);
        total_del_incorrect = np.append(total_del_incorrect,del_incorrect);
        #Plot
        fig = plt.figure();
        ax = fig.add_subplot(211);
        ax.hist(del_correct,bins=NSUB,range=[0, NSUB]);
        ax2 = fig.add_subplot(212);
        ax2.hist(del_incorrect,bins=NSUB,range=[0, NSUB]);
    #Plot all regions together
    fig = plt.figure();
    ax = fig.add_subplot(211);
    ax.hist(total_del_correct,bins=NSUB,range=[0, NSUB]);
    ax2 = fig.add_subplot(212);
    ax2.hist(total_del_incorrect,bins=NSUB,range=[0, NSUB]);
