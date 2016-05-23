# -*- coding: utf-8 -*-
"""
Created on Mon May 16 22:26:19 2016

@author: ReidW
"""

import numpy as np
from scipy import signal
from numpy import fft
from statsmodels.tsa import stattools
from scikits.talkbox import features
from scipy.stats import binned_statistic

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

    samples = s.getSubsamples();
    fs = s.waveparms.fs;
    #TODO: next power of 2?
    (M,L) = np.shape(s.samples);

    if win != None:
        #TODO: add windows support
        pass

    #Run
    F = np.empty([M,N]); #Array for storing feature output, N for each sample, M samples per supersample
    for i in range(M):
        isample = samples[i,:];

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
            lbase = 3.0;
            #bin_ends = np.logspace(np.log10(fax[2])/np.log10(lbase),np.log10(fax[-1])/np.log10(lbase),N+1,base=lbase)
            bin_ends = np.logspace(np.log10(10)/np.log10(lbase),np.log10(fax[-1])/np.log10(lbase),N+1,base=lbase)
        elif spacing=="lin":
            bin_ends = np.linspace(fax[1],fax[-1],N+1)
        bin_ends[0] = -1;

        for j in range(N): #loop over bins
            this_bin = (fax>bin_ends[j]) & (fax<=bin_ends[j+1]);
            if sum(np.round(this_bin))==0:
                print("Bad binning in FFT alg");
            if smooth_meth==smooth_method.peak:
                F[i,j] = np.max(P[this_bin])
            elif smooth_meth==smooth_method.mean:
                F[i,j] = np.mean(P[this_bin])
            elif smooth_meth==smooth_method.median:
                F[i,j] = np.median(P[this_bin])

    #TODO: if we want a frequency axis, use centers of bins. Current axis doesn't match data returned.
    return (fax,F);


def getAllSPED(sups, N, twin = 3, fwin = 21, nperseg=256, spacing="log"):
    nsup = np.shape(sups)[0];
    M = sups[0].N;

    F_all = np.empty([nsup,M,N]);
    for (i,s) in enumerate(sups):
        F_all[i,:,:] = getSupersampleSPED(s, N, twin, fwin, nperseg, spacing);

    return F_all;


def getSupersampleSPED(s, N, twin = 3, fwin = 21, nperseg=256, spacing="log"):
    samples = s.getSubsamples();
    #For simplicity, twin, fwin must be odd
    twin = int( 2*np.floor(twin/2) + 1);
    fwin = int( 2*np.floor(fwin/2) + 1);
    h_twin = np.floor(twin/2); h_fwin = np.floor(fwin/2);

    fs = s.waveparms.fs;
    (M,L) = np.shape(samples);

    BLOCKSIZE = 2**10

    F = np.empty([M,N]); #Array for storing feature output, N for each sample, M samples per supersample
    for i in range(M):
        isample = samples[i,:];
        #Get spectrogram of this sample
        fax, tax, spec = signal.spectrogram(isample,fs,nperseg=nperseg,nfft=BLOCKSIZE);

        (nf,nt) = np.shape(spec);
        isped = np.zeros((nf,nt));
        isped2 = np.zeros((nf,nt));
        isped3 = np.zeros((nf,nt));

        """
        for jf in range(nf):
            for jt in range(nt):
                if ( spec[jf,jt] >= spec[ max(0,jf-h_fwin):min(nf,jf+h_fwin),
                       max(0,jt-h_twin):min(nt,jt+h_twin) ] ).all():
                    isped[jf,jt] = 1;
        """


        num_fbin = int(np.ceil(nf/h_fwin));
        peaks = np.zeros((num_fbin,nt) );
        for j in range(nt):
            for jfbin in range( num_fbin ):
                fbin = spec[jfbin*h_fwin:min(nf,(jfbin+1)*h_fwin), j];
                peaks[jfbin,j] = np.amax( fbin );
        for j in range(nt):
            for jfbin in range( num_fbin ):
                if (peaks[jfbin,j] >= peaks[max(0,jfbin-1):min(num_fbin,jfbin+2),max(0,j-h_twin):min(nt,j+h_twin+1)]).all():
                    fbin = spec[jfbin*h_fwin:min(nf,(jfbin+1)*h_fwin), j];
                    isped3[jfbin*h_fwin:min(nf,(jfbin+1)*h_fwin),j] = np.round(fbin >=peaks[jfbin,j]);
        isped = isped3;
        """
        #Hopefully more efficient?
        num_fbin = int(np.ceil(nf/h_fwin));
        peaks = np.zeros((num_fbin,nt) ); peak_inds = np.zeros((num_fbin,nt));
        for j in range(nt):
            for jfbin in range( num_fbin ):
                fbin = spec[jfbin*h_fwin:min(nf,(jfbin+1)*h_fwin), j];
                peaks[jfbin,j] = np.amax( fbin );
                peak_inds[jfbin,j] = jfbin*h_fwin + np.argmax( fbin );
        for j in range(nt):
            for jpeak in range(num_fbin):
                f_window = (abs(peak_inds - peak_inds[jpeak,j])<=h_fwin);
                f_window[:,0:max(0,j-h_twin)] = False; f_window[:,min(nt,j+h_twin):nt] = False
                f_window[jpeak,j] = False;
                #t_window = np.full((num_fbin,nt),False,dtype=bool);t_window[:,max(0,j-h_twin):min(nt,j+h_twin+1)] = True;
                #if (peaks[jpeak,j]>= (peaks[ f_window & t_window]) ).all():
                if (peaks[jpeak,j]> (peaks[ f_window ]) ).all():
                    isped2[peak_inds[jpeak,j],j] = 1;
                    #if peak_inds[jpeak,j] > 0:
                    #    isped2[peak_inds[jpeak,j],j] = 1;
        isped = isped2;
        """

        ispedf = np.sum(isped,1);
        #ispedf = ispedf + 1; #Laplace smoothing?

        #Need to downsample to N bins
        if spacing=="log":
            lbase = 3.0;
            bin_ends = np.logspace(np.log10(fax[3])/np.log10(lbase),np.log10(fax[-1])/np.log10(lbase),N+1,base=lbase)
        elif spacing=="lin":
            bin_ends = np.linspace(fax[1],fax[-1],N+1)
        bin_ends[0] = -1;

        for j in range(N):
            isped_bin = ispedf[(fax>bin_ends[j]) & (fax<=bin_ends[j+1])];
            isped_bin.sort(); isped_bin[:] = isped_bin[::-1];

            F[i,j] = np.sum(isped_bin[0:min(5,np.size(isped_bin))]);

        F; #bp

    return F

def getSampleACF(sample, acflags ):
    acf = np.zeros((sample.Nsub,acflags+1))
    subsamples = sample.getSubsamples()
    for j,data in enumerate(subsamples):
        acf[j,:] = getSignalACF(data,acflags)
    return acf
    
def getSampleMFCC(sample, nceps=13):
    ceps = np.zeros((sample.Nsub,nceps))
    subsamples = sample.getSubsamples()
    for j,data in enumerate(subsamples):
        temp = features.mfcc(data,fs=sample.waveparms.fs,nceps=nceps)[0]
        temp[~np.isfinite(temp)] = 0            
        ceps[j,:] = np.mean(temp,0)
    return ceps
    
def getSignalACF(s,acflags=40):
    '''
    Binning causes problems. Seems that some of the bins are Infinite or NaN
    '''
    return stattools.acf(s,nlags=acflags,fft=True)
#    full_acf =  stattools.acf(s,nlags=len(s),fft=True)
#    return binned_statistic(full_acf,full_acf,bins=acflags+1)[0]
    
def getSignalMFCC(s,nceps=13,fs=44100):
    temp = features.mfcc(s,fs=fs,nceps=nceps)[0]
    temp[~np.isfinite(temp)] = 0
    return np.mean(temp,0)

