# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:28:42 2016

@author: ReidW
"""

import sys, os, datetime, re
import wave
import numpy as np
from scipy.io import wavfile
import csv

#IF NEEDED, run:
#  os.putenv('RECORDINGS_LOC',r'C:\Users\ReidW\Google Drive\Stanford\CS229\AudioLocator\Recordings')
#Get Folder for recordings
RECORD_DIR = os.environ.get('RECORDINGS_LOC');
if RECORD_DIR==None or not os.path.isdir(RECORD_DIR):
    RECORD_DIR = None
    print "Did not find 'RECORDINGS_LOC' home directory. Please Set."
csv_path = os.path.join(RECORD_DIR,'metadata.csv')

#TODO: Add normalization???

#Quick compile of recording filename regex
fn = re.compile('recording-(\d+)-(\d+)*');

#LABELS
#Quick enumeration of the regions, indexed from 0
class R:
    (Rains, Circle, Tresidder, Huang, Bytes, Oval, Arrillaga) = range(7);
Regions_dict = {"Rains":R.Rains,"Circle":R.Circle,"Tresidder":R.Tresidder,
           "Huang":R.Huang,"Bytes":R.Bytes,"Oval":R.Oval,"Arrillaga":R.Arrillaga};
REGIONS = [None]*7;
for jkey in Regions_dict.keys():
    REGIONS[Regions_dict[jkey]]=jkey;

class AudioParms:
    fs = None;
    nchannels = None;
    sampwidth = None;
    bitdepth = None;
    L = None; #length (in frames)

class Sample:
    waveparms = None;
    data=[];
    data_loaded = 0; #Don't really want to use
    L = 0; #Duration in total samples
    Nsub = 0; #Number of subsamples
    attrs = {};
    start_index = 0;

    def __init__(self,path, Lsub, Nsub, wp=None, start_index=None, get_attrs=False):
        '''
        Pass in the path to the containing file.
        To specify that this sample is only part of the whole file, pass in 'wp', a AudioParms() stuct
        Then, pass start_index, the index into the file that this sample starts.
        get_attrs = True: Read from metadata.csv for more information
        '''
        self.path = path;
        self.Lsub = Lsub;
        self.Nsub = Nsub;

        if wp!=None:
            self.waveparms = wp;
            self.L = wp.L;
        else:
            self.read_parms(); #Read them in if not provided
            self.L = wp.L;

        if start_index!=None:
            self.start_index = start_index; #Start index into file
        else:
            self.start_index = 0;

        (thisdir, self.filename) = os.path.split(self.path);
        res = fn.match(self.filename);
        self.date = datetime.date(int(res.groups()[0][0:4]), int(res.groups()[0][4:6]),
                                  int(res.groups()[0][6:8]));
        self.weekday = self.date.weekday(); #Monday = 0; Sunday = 6
        self.time = datetime.time(int(res.groups()[1][0:2]),int(res.groups()[1][2:4]),
                                  int(res.groups()[1][4:6]));

        #Use containing folder to get region (i.e. label);
        jregion = thisdir.split('\\')[-1]

        self.region = Regions_dict[jregion];

        #User file path and metadata.csv file to get additional data
        if get_attrs:
            self.get_attrs();

    def get_attrs(self):
        fname=os.path.splitext(self.filename)[0]
        with open(csv_path) as csvfile:
            mycsv = csv.reader(csvfile)
            for row in mycsv:
                if row[0]==fname:
                    self.attrs['phone'] = row[4];
                    break;

    def read_parms(self):
        wf = wave.open(self.path,'r');
        self.waveparms = AudioParms();
        self.waveparms.fs = self.wf.getframerate();
        self.waveparms.nchannels = self.wf.getnchannels();
        self.waveparms.sampwidth = self.wf.getsampwidth(); #Bytes
        self.waveparms.L = self.wf.getnframes();
        wf.close();

    def getSubsamples(self):
        #Return data as an array of subsamples
        if self.data_loaded:
            samp_data = np.empty((self.Nsub,self.Lsub));
            for isub in range(self.Nsub):
                samp_data[isub,:] = self.data[isub*self.Lsub:(isub+1)*self.Lsub];
        else:
            data = np.array(wavfile.read(self.path)[1],dtype=float);
            samp_data = np.empty((self.Nsub,self.Lsub));
            for isub in range(self.Nsub):
                if self.waveparms.nchannels == 2:
                    this_sub = data[self.start_index + isub*self.Lsub: self.start_index + (isub+1)*self.Lsub,0]
                else:
                    this_sub = data[self.start_index + isub*self.Lsub: self.start_index + (isub+1)*self.Lsub]
                samp_data[isub,:] = this_sub/(2**(self.waveparms.sampwidth*8-1));

        return samp_data
    def getData(self):
        #Return all data in this sample
        if self.data_loaded:
            return self.data;
        else:
            data = np.array(wavfile.read(self.path)[1],dtype=float);
            if self.waveparms.nchannels == 2:
                data = data[self.start_index: self.start_index + self.L,0];
            else:
                data = data[self.start_index: self.start_index + self.L];
            data = data/(2**(self.waveparms.sampwidth*8-1));
            return data;

"""
def getAllSamples(T=None,N=None,key=None,val=None):
    samples = [];
    i=0;
    for root, dirs, files in os.walk(RECORD_DIR):
        for jfile in files:
            if os.path.splitext(jfile)[1] == '.wav':
                this_sample = Supersample(os.path.join(root,jfile));
                if key!=None:
                    #Put a key here for getting only certain data
                    this_sample.get_attrs();
                    if this_sample.attrs[key] == val:
                        samples.append(this_sample);
                        #Add only if our key/val match
                        if T!=None:
                            samples[i].gather_samples(T,N);
                        i+=1;
                else:
                    samples.append(this_sample);
                    if T!=None:
                        samples[i].gather_samples(T,N);
                    i+=1;
    print "Imported %i Supersamples"%i
    return samples;
"""
def getAllSamples(Tsub=None,Nsub=None,key=None,val=None,READ_IN=True):
    samples = [];
    i=0;
    for root, dirs, files in os.walk(RECORD_DIR):
        for jfile in files:
            if os.path.splitext(jfile)[1] == '.wav':
                jpath = os.path.join(root,jfile);
                skip = 0
                if key=="phone":
                    fname=os.path.splitext(jfile)[0]
                    with open(csv_path) as csvfile:
                        mycsv = csv.reader(csvfile)
                        for row in mycsv:
                            if row[0]==fname and row[4] != val:
                                skip = 1
                                break
                            else:
                                break
                if skip == 1:
                    continue

                file_parms = AudioParms();
                wf = wave.open(jpath,'r')
                file_parms.fs = wf.getframerate();
                file_parms.nchannels = wf.getnchannels();
                file_parms.sampwidth = wf.getsampwidth();
                total_len = wf.getnframes();
                wf.close();

                Lsub = file_parms.fs*Tsub; #Length of subsample
                Lsamp = Lsub*Nsub;
                file_parms.L = Lsamp; #Length of each sample object
                Nsamp = int(np.floor(total_len/Lsamp));

                if READ_IN:
                    data = np.array(wavfile.read(jpath)[1],dtype=float);
                    for isample in range(Nsamp):
                        #Grab just first channel
                        if file_parms.nchannels == 2:
                            this_sample_data = data[isample*Lsamp:(isample+1)*Lsamp,0]
                        else:
                            this_sample_data = data[isample*Lsamp:(isample+1)*Lsamp]

                        this_sample = Sample(jpath, Lsub, Nsub, wp = file_parms, start_index = isample*Lsamp)
                        #Convert to float, normalized to [-1,1]
                        this_sample.data = this_sample_data/(2**(file_parms.sampwidth*8-1));
                        this_sample.data_loaded = 1;
                        samples.append(this_sample);
                        i+=1;
                else:
                    for isample in range(Nsamp):
                        this_sample = Sample(jpath, Lsub, Nsub, wp = file_parms, start_index = isample*Lsamp)
                        this_sample.data_loaded = 0;
                        samples.append(this_sample);
                        i+=1;

    print "Imported %i Supersamples"%i
    return samples;

def findSamples(key,val):
    #Find samples based on the value of a key, such as day of week, and time range
    #TODO
    pass

#wavefile = r'C:\Users\ReidW\Google Drive\Stanford\CS229\AudioLocator\Recordings\Arrillaga\recording-20160515-185508.wav'
#mysamp = Supersample(wavefile)
#mysamp.get_attrs()