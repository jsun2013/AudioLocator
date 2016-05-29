# -*- coding: utf-8 -*-
"""
Created on Sat May 28 09:56:54 2016

@author: ReidW
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 27 01:28:40 2016

@author: ReidW
"""
import os
import numpy as np
import pickle

import samples
reload(samples)
import spectral
reload(spectral)
import audiolearning
reload(audiolearning)
import mytimer as mt

#PARAMETERS
FWIN = 25;
TWIN = 1;
NPERSEG = 1024;
TSUB = 1;
NSUB = 10;
NFFT = 180; #Resolution used in fwdSearch
NSEL = 20; #Number of bins to use, capped out at number generated in fwdSearch

feat_file = "fwdSearch_%i_%i.pkl"%(NFFT,40);
#Hmmm, could later find first file with at least NSEL bins selected. Manual for now.
if not os.path.exists(feat_file):
    raise ValueError("This forward search file not found.")

data_file = 'reducedPhi_%i_%i_%i_%i_%i_%i_%i.pkl'%(TSUB,NSUB,NFFT,NSEL,FWIN,TWIN,NPERSEG);
LOAD_DATA = True;

#OTHER OPTIONS
iters = 20;
#fwdSearch file to use

frac_test = 0.2;

class reducedPhi:
    LEN = 0;
    def __init__(self):
        self.LEN = NSEL+13;
        with open(feat_file,'rb') as myPkl:
            fwdSearch = pickle.load(myPkl)
        keep_inds = fwdSearch['bin_inds'];
        keep_inds = keep_inds[:NSEL];
        keep_inds.sort()
        self.keep_inds = keep_inds;

    def get_phi(self,sample):
        '''
        Takes in a super_sample and returns a feature array. Breaks the super_sample
        down into samples. Each row of the returned value corresponds to a sample
        in the super sample.
        '''

        #XSPED = spectral.getSupersampleSPED(sample,self.fft_bins,fwin=FWIN,twin = TWIN,nperseg = NPERSEG,spacing="log")
        XSPED = spectral.getSupersampleSPEDselect(sample,NFFT,self.keep_inds,fwin=FWIN,twin = TWIN,nperseg = NPERSEG,spacing="log")
        XMFCC = spectral.getSampleMFCC(sample)
        return np.hstack((XMFCC,XSPED))
        #return XSPED

if __name__ == "__main__":
    mt.tic()


    #Create a classifier instance with your feature extraction method
    myPhi = reducedPhi();
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

    totalErr = 0;
    for i_iter in range(iters):
        #Shuffle up the order
        inds = range(nsamp);
        np.random.shuffle(inds);

        X_train = X[inds[num_test:],:,:];
        Y_train = Y[inds[num_test:]];

        X_test = X[inds[:num_test],:,:];
        Y_test = Y[inds[:num_test]];

        extractor.trainSVMBatch(train_samples=None, X_train=X_train, Y_train=Y_train,kernel='rbf',
                                    C=50000,gamma=1/(10000*float(myPhi.LEN)) );

        j_totalErr = extractor.testClassifier(test_samples=None,X_test=X_test,
                                            Y_test=Y_test);
        totalErr += j_totalErr;

    totalErr = totalErr/iters;
    print("\n\nMean Test Error = %0.06f"%totalErr)
