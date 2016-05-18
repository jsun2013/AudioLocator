# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:28:42 2016

@author: ReidW
"""

import sys, os, datetime, re
import wave
import numpy as np
from scipy.io import wavfile

#IF NEEDED, run:
#  os.putenv('RECORDINGS_LOC',r'C:\Users\ReidW\Google Drive\Stanford\CS229\AudioLocator\Recordings')
#Get Folder for recordings
RECORD_DIR = os.environ.get('RECORDINGS_LOC');
if RECORD_DIR==None or not os.path.isdir(RECORD_DIR):
    RECORD_DIR = None
    print "Did not find 'RECORDINGS_LOC' home directory. Please Set."

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
    bitdepth = None;
    L = None; #length (in frames)

class Supersample:
    #Contains information about the supersample collected
#    self.path=None;
#    self.filename = None;
#    self.date = None;
#    self.day = None;
#    self.time = None;
#    self.region = None;
#    self.location = None;
#    self.add_notes = None;
    wf = None;
    waveparms = None;
    data = None; data_read = 0;

    def __init__(self,path,get_attrs=False):
        self.path = path;
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

        #Additional attributes contained in the metadata spreadsheet
        if get_attrs:
            self.get_attrs();


    def get_attrs(self):
        #Reads in the metadata file to find its attributes

        #If CSV mismatches previous set for region (from folder), complain
        pass


    def read_parms(self):
        self.wf = wave.open(self.path,'r');
        self.waveparms = AudioParms();
        self.waveparms.fs = self.wf.getframerate();
        self.waveparms.nchannels = self.wf.getnchannels();
        self.waveparms.sampwidth = self.wf.getsampwidth(); #Bytes
        self.waveparms.L = self.wf.getnframes();
        self.wf.close();

    def read_data(self,start = None, length = None, unit = None):
        if self.waveparms == None:
            self.read_parms();

        if self.data_read == 0:
            #Read in entire file
            self.data = np.array(wavfile.read(self.path)[1],dtype=float);
            #Normalize to [-1,1
            """if self.waveparms.nchannels == 2:
                self.data = np.array([
                        self.data[1::2]/(2**(self.waveparms.sampwidth*8-1)),
                        self.data[2::2]/(2**(self.waveparms.sampwidth*8-1))  ])
            else:"""
            self.data = np.array(self.data/(2**(self.waveparms.sampwidth*8-1)));
            self.data_read = 1;

        if start != None and length != None:
            if unit=='sec':
                start = start*self.waveparms.fs;
                length = length*self.waveparms.fs;
            this_data = self.data[start:start+length];
        else:
            this_data = self.data;

        #FOR NOW, ONLY RETURNING FIRST CHANNEL.
        if self.waveparms.nchannels == 2:
            this_data = this_data[:,0];

        return this_data;

    def close_data(self):
        #For the purpose of saving memory,
        self.data = None; self.data_read = 0;

def getAllSamples():
    samples = [];
    for root, dirs, files in os.walk(RECORD_DIR):
        for jfile in files:
            if os.path.splitext(jfile)[1] == '.wav':
                samples.append(Supersample(os.path.join(root,jfile)));
    return samples;


def findSamples(key,val):
    #Find samples based on the value of a key, such as day of week, and time range
    #TODO
    pass