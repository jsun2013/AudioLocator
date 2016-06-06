# -*- coding: utf-8 -*-
"""
Created on Thu Jun 02 13:41:27 2016

@author: ReidW
"""

import os, sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import samples as s
reload(s)
import spectral
reload(spectral)
import audiolearning
reload(audiolearning)

RECORD_DIR = s.RECORD_DIR;

region1 = 'Arrillaga';
region2 = region1;
recording1 = '20160516-151425'; #Weights
rec1_title = "Monday, Arrillaga"
recording2 = '20160515-185831'; #Weights
rec2_title = "Sunday, Arrillaga"

region3 = 'Circle';
region4 = 'Bytes';
recording3 = '20160525-181302';
rec3_title = "Circle"
recording4 = '20160531-111802';
rec4_title = "Bytes"

file1_path = os.path.join(RECORD_DIR,region1+"//recording-"+recording1+".wav");
file2_path = os.path.join(RECORD_DIR,region2+"//recording-"+recording2+".wav");
file3_path = os.path.join(RECORD_DIR,region3+"//recording-"+recording3+".wav");
file4_path = os.path.join(RECORD_DIR,region4+"//recording-"+recording4+".wav");

#PARAMETERS
FFT_BINS = 60;
FWIN = 25;
TWIN = 1;
NPERSEG = 1024;
TSUB = 1; #NOTE: this will be used only for training. Will experiment with test format later.
NSUB = 60; #This is just used for feature generation, doesn't really matter.
TEST_NSUB = 10;

samp1 = s.getSample(file1_path,TSUB,NSUB);
samp2 = s.getSample(file2_path,TSUB,NSUB);
samp3 = s.getSample(file3_path,TSUB,NSUB);
samp4 = s.getSample(file4_path,TSUB,NSUB);

fs=samp1.waveparms.fs;
subsamp = 10; #Doesn't really matter, just be consistent
data1 = samp1.getSubsamples(subsamp); #get last subsample
data2 = samp2.getSubsamples(subsamp); #get last subsample
data3 = samp3.getSubsamples(subsamp); #get last subsample
data4 = samp4.getSubsamples(subsamp); #get last subsample

#SPECTROGRAM
f, t, S1 = signal.spectrogram(data1,fs);
f, t, S2 = signal.spectrogram(data2,fs);
f, t, S3 = signal.spectrogram(data3,fs);
f, t, S4 = signal.spectrogram(data4,fs);

plt.figure();
plt.subplot(1,4,1);
plt.pcolormesh(t,f,np.log10(S1));
plt.ylim((0,22000));
plt.ylabel('Frequency (Hz)',fontsize=20);
plt.xlabel('Time (sec)',fontsize=14);
plt.title(rec1_title);

ax = plt.subplot(1,4,2);
plt.pcolormesh(t,f,np.log10(S2));
ax.set_yticklabels([]);
plt.ylim((0,22000));
plt.xlabel('Time (sec)',fontsize=14);
plt.title(rec2_title);

ax = plt.subplot(1,4,3);
plt.pcolormesh(t,f,np.log10(S3));
ax.set_yticklabels([]);
plt.ylim((0,22000));
plt.xlabel('Time (sec)',fontsize=14);
plt.title(rec3_title);

ax = plt.subplot(1,4,4);
plt.pcolormesh(t,f,np.log10(S4));
ax.set_yticklabels([]);
plt.ylim((0,22000));
plt.xlabel('Time (sec)',fontsize=14);
plt.title(rec4_title);

sped1 = spectral.getSupersampleSPED(samp1, FFT_BINS, twin=TWIN, fwin=FWIN, nperseg=NPERSEG)
sped2 = spectral.getSupersampleSPED(samp2, FFT_BINS, twin=TWIN, fwin=FWIN, nperseg=NPERSEG)
sped3 = spectral.getSupersampleSPED(samp3, FFT_BINS, twin=TWIN, fwin=FWIN, nperseg=NPERSEG)
sped4 = spectral.getSupersampleSPED(samp4, FFT_BINS, twin=TWIN, fwin=FWIN, nperseg=NPERSEG)
sped1 = np.divide(np.sum(sped1,0),samp1.Nsub);
sped2 = np.divide(np.sum(sped2,0),samp1.Nsub);
sped3 = np.divide(np.sum(sped3,0),samp1.Nsub);
sped4 = np.divide(np.sum(sped4,0),samp1.Nsub);

plt.figure();
#plt.title("Arrillaga",fontsize=28)
plt.bar(np.arange(0,FFT_BINS,1),sped1,width=0.4,color="r");
plt.bar(np.arange(0.4,FFT_BINS+0.4,1),sped2,width=0.4);
plt.xlabel("Frequency Bin",fontsize=20);
#plt.ylabel("Peak Count",fontsize=24);
plt.ylim((0,50))
#plt.legend(loc=9);

plt.figure();
plt.bar(np.arange(0,FFT_BINS,1),sped3,width=0.4,color="r",label=rec3_title);
plt.bar(np.arange(0.4,FFT_BINS+0.4,1),sped4,width=0.4,label=rec4_title);
plt.xlabel("Frequency Bin Number",fontsize=20);
#plt.ylabel("Peak Count",fontsize=24);
plt.ylim((0,50))
plt.legend(loc=9);