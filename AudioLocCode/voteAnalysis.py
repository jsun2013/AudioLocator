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
PLOT=False

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
    vote_rec_per = [[np.empty((0,2)), np.empty((0,3))] for i in range(7)]
    totalErr = 0
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

        (jtotalErr,test_actual,test_hat,vote_rec) = extractor.testClassifier(test_samples=None,X_test=X_test,
                                            Y_test=Y_test,get_conf_mat=False,return_rec=True);
        totalErr+=jtotalErr;
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
                #hit1 = max(this_count); #Winning vote.
                hit0 = this_count[test_hat[i]]; #Winning vote
                this_count[test_hat[i]]=0;
                hit1 = max(this_count); #Runner up vote.
                if jlabel >= np.size(this_count):
                    hit2 = 0;
                else:
                    hit2 = this_count[jlabel]; #Actual label
                #vote_rec_per[jlabel][1].append([hit0, hit1]);
                vote_rec_per[jlabel][1] = np.append(vote_rec_per[jlabel][1],np.array([[hit0, hit1, hit2]]),axis=0);
    totalErr = totalErr/iters;
    print "Average test error = %0.02f%%"%(100*totalErr);
    total_del_correct = np.empty(0);
    total_del_incorrect = np.empty(0);
    total_incorrect_deficit = np.empty(0);
    for ireg in range(7):
        del_correct = vote_rec_per[ireg][0][:,0] - vote_rec_per[ireg][0][:,1];
        del_incorrect = vote_rec_per[ireg][1][:,0] - vote_rec_per[ireg][1][:,1];
        incorrect_deficit = vote_rec_per[ireg][1][:,0] - vote_rec_per[ireg][1][:,2];
        #Add to overall total
        total_del_correct = np.append(total_del_correct,del_correct);
        total_del_incorrect = np.append(total_del_incorrect,del_incorrect);
        total_incorrect_deficit = np.append(total_incorrect_deficit,incorrect_deficit);
        #TODO: pyplot hist is garbage. Try manual binning and step plot.
        #Plot
        if PLOT:
            fig = plt.figure();
            ax = fig.add_subplot(211);
            ax.hist(del_correct,range=(0,NSUB),align='right');
            ax2 = fig.add_subplot(212);
            ax2.hist(del_incorrect,range=(0,NSUB),bins=NSUB,align='right');
    #Gather some voting statistics
    n_incorrect = np.size(total_del_incorrect); n_correct = np.size(total_del_correct);

    #Percentage of incorrect predictions when the vote-gap was 0, 1 respectively
    n_del1m_correct = np.sum(np.round(total_del_correct<=1));
    n_del1m_incorrect = np.sum(np.round(total_del_incorrect<=1));
    n_del1_from_correct = np.sum(np.round(total_incorrect_deficit<=1));
    print "%0.02f%% of samples have margin <=1"%(100*(n_del1m_correct+n_del1m_incorrect)/(n_correct+n_incorrect))

    perc_tie_err = np.sum(np.round(total_del_incorrect==0))/n_incorrect;
    perc_del1_err = np.sum(np.round(total_del_incorrect==1))/n_incorrect;
    print "%% of incorrect prediction coming from small voting margins:"
    print "\t%0.02f%% Ties"%(100*perc_tie_err)
    print "\t%0.02f%% Margin=1"%(100*perc_del1_err)
    print "\t%0.02f%% Margin<=1"%(100*( perc_del1_err+perc_tie_err))

    perc_del1m_correct = n_del1m_correct/(n_del1m_correct + n_del1m_incorrect)
    print "%0.02f%% of samples with <=1 margin predicted correctly"%(100*perc_del1m_correct)
    perc_del1m_incorrect_close = n_del1_from_correct/(n_del1m_correct + n_del1m_incorrect)
    print "%0.02f%% of samples with <=1 margin predicted incorrectly with correct choice second"%(100*perc_del1m_incorrect_close)
    print "This is %0.02f%% error (which could be reduced)"%(100*n_del1_from_correct/(n_correct+n_incorrect))
    print "\t->this is %0.02f%% of our errors"%(100*n_del1_from_correct/n_incorrect)
    #Plot all regions together
    if PLOT:
        fig = plt.figure();
        ax = fig.add_subplot(311);
        ax.hist(total_del_correct,bins=NSUB+1,align='right');
        ax2 = fig.add_subplot(312);
        ax2.hist(total_del_incorrect,bins=NSUB+1,align='right');
        ax3 = fig.add_subplot(313);
        ax3.hist(total_incorrect_deficit,bins=NSUB+1,align='right');
