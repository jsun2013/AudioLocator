# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:26:55 2016

@author: ReidW
"""

import numpy as np
import pickle

import samples
reload(samples)
import spectral
reload(spectral)

class phi1:
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

        XSPED = spectral.getSupersampleSPED(sample,self.fft_bins,fwin=41,twin = 1,nperseg = 1024,spacing="log")
        return XSPED

FFT_BINS = 60;
myPhi = phi1(FFT_BINS);

all_samples = samples.getAllSamples(Tsub=2,Nsub=10,READ_IN=False);
subsamp_per = all_samples[0].Nsub;
nsamp = np.size(all_samples)

spedPhi = samples.DataPhi();
spedPhi.X = np.zeros((subsamp_per*nsamp,myPhi.LEN));
spedPhi.Y = np.zeros(subsamp_per*nsamp,dtype=np.int8);

print("Running feature extraction...")
nupdate = int(len(all_samples)/20);
k=0;
for (i,sample) in enumerate(all_samples):
    if i%nupdate==0:
        print("%d percent..."%((100*i)/nsamp));
    phi_X = myPhi.get_phi(sample)
    numSamples,_ = phi_X.shape
    spedPhi.X[k:k+numSamples,:] = phi_X
    spedPhi.Y[k:k+numSamples] = sample.region
    k += numSamples

print("Finished feature extraction.");

with open("spedPhi_2_10_60_45_1_1024.pkl","wb") as pklFile:
    pickle.dump(spedPhi,pklFile);