# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:45:03 2016

@author: ReidW
"""
import os, sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numpy import fft

import samples as s
reload(s)
import spectral
reload(spectral)
import audiolearning
reload(audiolearning)

#Comparison of bytes cafe
RECORD_DIR = s.RECORD_DIR;

region = 'Arrillaga';
if region == 'Bytes':
    recording1 = '20160516-143226'; #Ordering counter
    rec1_title = "Phone1 Monday, Location 1"
    recording2 = '20160516-143032'; #TV
    rec2_title = "Phone1 Monday, Location 2"

    recording3 = '20160515-153310'; #Ordering counter
    rec3_title = "Phone2 Sunday, Location 1"
    recording4 = '20160515-152744'; #TV
    rec4_title = "Phone2 Sunday, Location 2"
elif region == 'Tresidder':
    recording1 = '20160516-135739'; #Starbucks
    rec1_title = "Phone1 Monday, Location 1"
    recording2 = '20160516-140121'; #Fraiche
    rec2_title = "Phone1 Monday, Location 2"

    recording3 = '20160515-143003'; #Starbucks
    rec3_title = "Phone2 Sunday, Location 1"
    recording4 = '20160515-142349'; #Fraiche
    rec4_title = "Phone2 Sunday, Location 2"
elif region == 'Arrillaga':
    recording1 = '20160516-151425'; #Weights
    rec1_title = "Phone1 Monday, Location 1"
    recording2 = '20160516-150944'; #Fraiche
    rec2_title = "Phone1 Monday, Location 2"

    recording3 = '20160515-185831'; #Weights
    rec3_title = "Phone2 Sunday, Location 1"
    recording4 = '20160515-190142'; #Cardio
    rec4_title = "Phone2 Sunday, Location 2"

file1_path = os.path.join(RECORD_DIR,region+"//recording-"+recording1+".wav"); #Reid
file2_path = os.path.join(RECORD_DIR,region+"//recording-"+recording2+".wav");
file3_path = os.path.join(RECORD_DIR,region+"//recording-"+recording3+".wav"); #James
file4_path = os.path.join(RECORD_DIR,region+"//recording-"+recording4+".wav");

(Tsub, Nsub) = (2,5);
samp1 = s.Sample(file1_path,None,Nsub,Tsub=Tsub)
samp2 = s.Sample(file2_path,None,Nsub,Tsub=Tsub)
samp3 = s.Sample(file3_path,None,Nsub,Tsub=Tsub)
samp4 = s.Sample(file4_path,None,Nsub,Tsub=Tsub)

subsamp = 3; #Doesn't really matter, just be consistent
data1 = samp1.getSubsamples(subsamp); #get last subsample
data2 = samp2.getSubsamples(subsamp); #get last subsample
data3 = samp3.getSubsamples(subsamp); #get last subsample
data4 = samp4.getSubsamples(subsamp); #get last subsample

#Parameters
fs=samp1.waveparms.fs;
L =samp1.Lsub;

#STRAIGHT FFT
fax = fft.fftfreq(L,1/np.float(fs));
P1 = fft.fft(data1);
P3 = fft.fft(data3);
fax = fax[:L/2]; P1 = P1[:L/2]; P3 = P3[:L/2];

plt.figure();
plt.title(region,fontsize=28);
plt.semilogy(fax, np.abs(P1), label=rec1_title);
plt.semilogy(fax, np.abs(P3), color="r",label=rec3_title);
plt.xlabel("Frequency (Hz))");
plt.ylabel("Response, logscale");
plt.legend();
plt.tight_layout();


#SPED
fbin = 60; nperseg = 1024;
sped1 = spectral.getSupersampleSPED(samp1, fbin, twin=1, fwin=41, nperseg=nperseg)
sped2 = spectral.getSupersampleSPED(samp2, fbin, twin=1, fwin=41, nperseg=nperseg)
sped3 = spectral.getSupersampleSPED(samp3, fbin, twin=1, fwin=41, nperseg=nperseg)
sped4 = spectral.getSupersampleSPED(samp4, fbin, twin=1, fwin=41, nperseg=nperseg)
sped1 = np.sum(sped1,0);
sped2 = np.sum(sped2,0);
sped3 = np.sum(sped3,0);
sped4 = np.sum(sped4,0);

plt.figure();
plt.suptitle(region,fontsize=28);
plt.subplot(2,1,1);
plt.bar(np.arange(0,fbin,1),sped1,width=0.4,color="r",label=rec1_title);
plt.bar(np.arange(0.4,fbin+0.4,1),sped3,width=0.4,label=rec3_title);
#plt.xlabel("Frequency Bin Number");
plt.ylabel("Peak Count");
plt.legend(loc=9);

plt.subplot(2,1,2);
plt.bar(np.arange(0,fbin,1),sped2,width=0.4,color="r",label=rec2_title);
plt.bar(np.arange(0.4,fbin+0.4,1),sped4,width=0.4,label=rec4_title);
plt.xlabel("Frequency Bin Number"); plt.ylabel("Peak Count");
plt.legend(loc=9);


#SPECTROGRAM
f, t, S1 = signal.spectrogram(data1,fs);
f, t, S2 = signal.spectrogram(data2,fs);
f, t, S3 = signal.spectrogram(data3,fs);
f, t, S4 = signal.spectrogram(data4,fs);

plt.figure();
plt.suptitle(region,fontsize=28);
plt.subplot(1,4,1);
plt.pcolormesh(t,f,np.log10(S1));
plt.ylabel('Frequency (Hz)');
plt.xlabel('Time (sec)');
plt.title(rec1_title);

ax = plt.subplot(1,4,2);
plt.pcolormesh(t,f,np.log10(S2));
ax.set_yticklabels([]);
plt.xlabel('Time (sec)');
plt.title(rec2_title);

ax = plt.subplot(1,4,3);
plt.pcolormesh(t,f,np.log10(S3));
ax.set_yticklabels([]);
plt.xlabel('Time (sec)');
plt.title(rec3_title);

ax = plt.subplot(1,4,4);
plt.pcolormesh(t,f,np.log10(S4));
ax.set_yticklabels([]);
plt.xlabel('Time (sec)');
plt.title(rec4_title);


