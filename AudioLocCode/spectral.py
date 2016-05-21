# -*- coding: utf-8 -*-
"""
Created on Mon May 16 22:26:19 2016

@author: ReidW
"""

import numpy as np
from scipy import signal
from numpy import fft

"""
%Example usage:

%%%%%%%%%%%%%
%Single supersample:
import samples as s
import spectral
wavefile = r'C:\Users\ReidW\Google Drive\Stanford\CS229\AudioLocator\Recordings\Arrillaga\recording-20160516-150405.wav'
mysamp = s.Supersample(wavefile);
mysamp.gather_samples(T=5); %Individual samples 5 seconds long
F = getSupersampleFFT(mysamp, 40, smooth_meth = smooth_method.peak);

%%%%%%%%%%%%%
%Entire Dataset:
import samples as s
import spectral
sups = s.getAllSamples(T=10,N=2) #10 second samples, 2 samples per supersample
F_all = spectral.getAllFFT(sups, 40)
%F_all[i,j,:] gives spectrum for the ith supersample, jth sample
"""


#Used to enumerate supported windows
class windows:
    blackman = 'blackman'
    hamming = 'hamming'
    hann = 'hann'

class spectral_method:
    (periodogram, fft, welch) = range(3);

class smooth_method:
    (peak,mean,medin) = range(3);

def getAllFFT(sups, N, spec_meth = spectral_method.periodogram,
                smooth_meth = smooth_method.mean, spacing = "log", win=None, overlap=0):
    nsup = np.shape(sups)[0];
    (M,L) = np.shape(sups[0].samples);

    F_all = np.empty([nsup,M,N]);
    for (i,s) in enumerate(sups):
        (fax, F_all[i,:,:]) = getSupersampleFFT(s, N, spec_meth, smooth_meth, spacing, win, overlap);

    return F_all;

def getSupersampleFFT(s, N, spec_meth = spectral_method.periodogram,
                      smooth_meth = smooth_method.mean, spacing = "log", win=None, overlap=0):
    '''
    s is one Supersample object
    N is number of frequency bins to return
    spec_meth = method for finding spectrum (from class spectral_method)
    smooth_meth = method for downsampling spectrum to N bins (from class smooth_method)
    spacing = spacing of output bins. Either "log" or "lin"
    win = choice from windows class. E.g. windows.hamming
    overlap = fraction of overlap used when windowing

    if N is specified, it will be used over T
    '''
    if s.samples_loaded == 0:
        #Data has not been read in yet
        #TODO: read automatically?
        raise ValueError("Error: sample data not read in yet before getSupersampleFFT()")

    fs = s.waveparms.fs;
    #TODO: next power of 2?
    (M,L) = np.shape(s.samples);

    if win != None:
        #TODO: add windows support
        pass

    #Run
    F = np.empty([M,N]); #Array for storing feature output, N for each sample, M samples per supersample
    for i in range(M):
        isample = s.samples[i,:];

        #Use specified method to get a spectrum
        if spec_meth==spectral_method.periodogram:
            (fax, P) = signal.periodogram(isample,fs)
        elif spec_meth==spectral_method.fft:
            fax = fft.fftfreq(L,1/np.float(fs));
            P = fft.fft(isample);
        elif spec_meth==spectral_method.welch:
            fax, P = signal.welch(isample, fs, scaling='spectrum')

        #Now have frequency axis and spectrum. Need to downsample to N bins
        if spacing=="log":
            bin_ends = np.logspace(0,np.log10(fax[-1]),N+1)
        elif spacing=="lin":
            bin_ends = np.linspace(fax[1],fax[-1],N+1)
        bin_ends[0] = -1;

        for j in range(N): #loop over bins
            if smooth_meth==smooth_method.peak:
                F[i,j] = np.max(P[(fax>bin_ends[j]) & (fax<=bin_ends[j+1])])
            elif smooth_meth==smooth_method.mean:
                F[i,j] = np.mean(P[(fax>bin_ends[j]) & (fax<=bin_ends[j+1])])
            elif smooth_meth==smooth_method.median:
                F[i,j] = np.median(P[(fax>bin_ends[j]) & (fax<=bin_ends[j+1])])

    return (fax,F);
#import matplotlib.pyplot as plt

