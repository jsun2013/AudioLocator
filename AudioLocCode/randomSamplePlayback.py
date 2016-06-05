# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 08:46:24 2016

@author: ReidW
"""
import os
import pyaudio
import random
import numpy as np
import samples as s
import wave
#For convenience, read in as 10 second chunks for playback
TSUB = 10;
NSUB = 1;
CHUNK = 1024;

LEN = 10; #seconds

#Import all samples, but don't read in audio yet
readback_samples = s.getAllSamples(Tsub=TSUB,Nsub=NSUB,READ_IN=False) #
NSAMP = len(readback_samples);

def playRandomSample(loop=False):
    i = random.randint(1,NSAMP)-1;
    this_samp = readback_samples[i];
    this_region = this_samp.region; #Integer value
    #Waveforem parameters
    wf = this_samp.waveparms;

    #TODO: Do we want to open this outside, so we aren't calling this and terminate all the time?
    p = pyaudio.PyAudio();

    stream = p.open(format=p.get_format_from_width(wf.sampwidth),
                    channels=wf.nchannels, #Just do mono for all
                    rate=wf.fs,
                    output=True)

    #Read in all 10 seconds of data
    data_len = LEN * wf.fs;
    ind = 0;
    #with wave.open(this_samp.path,'rb') as mywave:
    mywave =  wave.open(this_samp.path,'rb')
    #seek to this spot
    mywave.setpos(this_samp.start_index);
    data = mywave.readframes(CHUNK);
    while(ind < data_len-1):
        stream.write(data);
        ind+=CHUNK;
        data = mywave.readframes(CHUNK);
        #TAKE INTERRUPT HERE

    stream.stop_stream()
    stream.close()

    p.terminate()


    return this_region

if __name__ == "__main__":
    playRandomSample()