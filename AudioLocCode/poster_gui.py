# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 10:46:47 2016

@author: james
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 08:30:08 2016

@author: james
"""

import sys
import os
import numpy as np
import pickle
import pyaudio
import random
import wave
import time

from sklearn import linear_model,svm,ensemble,tree
import samples as s
reload(s)
import spectral
reload(spectral)
import audiolearning
reload(audiolearning)
import mytimer as mt
reload(mt)
import playbackTestTrack as ptt
reload(ptt)

from PyQt4 import QtGui

class phi_sped_mfcc:
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

        XSPED = spectral.getSupersampleSPED(sample,self.fft_bins,fwin=25,twin = 1,nperseg = 1024,spacing="log")
        XMFCC = spectral.getSampleMFCC(sample)
        return np.hstack((XMFCC,XSPED))


class Example(QtGui.QWidget):
    def __init__(self):
        super(Example, self).__init__()
        self.audio_int = False
        self.playing = False
        self.actual = None
        self.played_sample = None
    
        ''' Set Up Audio '''
        #For convenience, read in as 10 second chunks for playback
        self.TSUB = 1;
        self.NSUB = 10;
        self.CHUNK = 1024;
        
        self.LEN = 10; #seconds
        #Import all samples, but don't read in audio yet
        self.readback_samples = s.getAllSamples(Tsub=self.TSUB,Nsub=self.NSUB,READ_IN=False) #
        self.NSAMP = len(self.readback_samples);
        
        ''' Set up Classifier '''        
        self.phi = phi_sped_mfcc(60)   
        self.clf = None
        with open('Poster_clf.pkl','rb') as f:
            self.clf = pickle.load(f)
            print("Classifier Loaded")
        
        self.initUI()
        
    def playData(self):
        if self.played_sample is not None:
            this_samp = self.played_sample    
            self.p = pyaudio.PyAudio();
            wf = this_samp.waveparms;
        
        
            stream = self.p.open(format=self.p.get_format_from_width(wf.sampwidth),
                            channels=wf.nchannels, #Just do mono for all
                            rate=wf.fs,
                            output=True)
        
            #Read in all 10 seconds of data
            data_len = self.LEN * wf.fs;
            ind = 0;
            #with wave.open(this_samp.path,'rb') as mywave:
            mywave =  wave.open(this_samp.path,'rb')
            #seek to this spot
            mywave.setpos(this_samp.start_index);
            data = mywave.readframes(self.CHUNK);
            self.playing = True
            while(ind < data_len-1) and not self.audio_int:
                stream.write(data);
                ind+=self.CHUNK;
                data = mywave.readframes(self.CHUNK);
                #TAKE INTERRUPT HERE
            self.playing = False
            stream.stop_stream()
            stream.close()      
        
    def playRandomSample(self,loop=False,region=None):
        self.p = pyaudio.PyAudio();
        if region is None:
            i = random.randint(1,self.NSAMP)-1
        else:
            possible_ind = [x for x,sample in enumerate(self.readback_samples) if sample.region==region ]
            i = np.random.choice(possible_ind)
        this_samp = self.readback_samples[i];
        this_region = this_samp.region; #Integer value
        #Waveforem parameters
        wf = this_samp.waveparms;
    
    
        stream = self.p.open(format=self.p.get_format_from_width(wf.sampwidth),
                        channels=wf.nchannels, #Just do mono for all
                        rate=wf.fs,
                        output=True)
    
        #Read in all 10 seconds of data
        data_len = self.LEN * wf.fs;
        ind = 0;
        #with wave.open(this_samp.path,'rb') as mywave:
        mywave =  wave.open(this_samp.path,'rb')
        #seek to this spot
        mywave.setpos(this_samp.start_index);
        data = mywave.readframes(self.CHUNK);
        self.playing = True
        while(ind < data_len-1) and not self.audio_int:
            stream.write(data);
            ind+=self.CHUNK;
            data = mywave.readframes(self.CHUNK);
            #TAKE INTERRUPT HERE
        self.playing = False
        stream.stop_stream()
        stream.close()
        self.p.terminate()
    
    
        return (this_region,this_samp)
        
    def regionCallback(self):
        choice = self.sender().text()
        for i in range(7):
            self.regionBtns[i].setEnabled(False)
        if self.practicCheck.isChecked():
            self.playRandomSample(region=s.Regions_dict[choice])
            print "played: ", choice
        else:
            phi_x,_ = self.clf.extract_features([self.played_sample])
            pred = self.clf.make_batch_prediction(phi_x,None)
            pred = int(pred[0])
            guess = s.Regions_dict[choice]
            test = s.REGIONS
            
            self.guess_label.setText(choice)
            self.pred_label.setText(s.REGIONS[pred])
            time.sleep(1)
            self.correct_label.setText(s.REGIONS[self.actual])
            ptt.addEntry(guess,pred,self.actual)
            (user_corr,class_corr) = ptt.getStats()
            self.human_label.setText(str(user_corr))
            self.machine_label.setText(str(class_corr))
            
            self.goBtn.setEnabled(True)
            self.played_sample = None
        

        for i in range(7):
            self.regionBtns[i].setEnabled(True)
        
    def practiceCallback(self):
        ''' This is a checkbox'''
            
        self.guess_label.setText('')
        self.pred_label.setText('')
        self.correct_label.setText('')
        if self.practicCheck.isChecked():
            self.goBtn.setEnabled(False)
            for i in range(7):
                self.regionBtns[i].setEnabled(True)
        else:
            self.goBtn.setEnabled(True)
            for i in range(7):
                self.regionBtns[i].setEnabled(False)
        
        
    def goCallback(self):
            
        self.guess_label.setText('')
        self.pred_label.setText('')
        self.correct_label.setText('')
        for i in range(7):
            self.regionBtns[i].setEnabled(True)
        self.actual,self.played_sample = self.playRandomSample()
        self.goBtn.setEnabled(False)
        self.restart_btn.setEnabled(True)
        
        
    def guessCallback(self):
        pass
        
    def initUI(self):
        
        self.setGeometry(300, 300, 1920,1080 )
        
        self.center()
        self.setWindowTitle('ABLoc')
        self.setWindowIcon(QtGui.QIcon('web.png')) 
        
        title = QtGui.QHBoxLayout()
        title.addStretch(1)
        title.addWidget(QtGui.QLabel('ABLoc'))
        title.addStretch(1)
        
        self.infoLabel = QtGui.QLabel('')
        self.practicCheck = QtGui.QCheckBox('Practice Mode',self) 
        self.goBtn = QtGui.QPushButton('Start!',self)  
        practiceLabel = QtGui.QLabel('')  
        
        (user_corr,class_corr) = ptt.getStats()
        self.human_label = QtGui.QLabel(str(user_corr))
        self.machine_label = QtGui.QLabel(str(class_corr))
        self.correct_label = QtGui.QLabel('')
        self.pred_label = QtGui.QLabel('')
        self.guess_label = QtGui.QLabel('')
        self.restart_btn = QtGui.QPushButton('Listen Again')
        self.restart_btn.setEnabled(False)
        
        
        grid = QtGui.QGridLayout()
        
        grid.setSpacing(10)
        self.practicCheck.stateChanged.connect(self.practiceCallback)
        self.goBtn.clicked.connect(self.goCallback)
        self.restart_btn.clicked.connect(self.playData)
        
        grid.addWidget(practiceLabel,1,0)
        grid.addWidget(self.practicCheck,1,1)
        grid.addWidget(practiceLabel,1,2)
        grid.addWidget(self.goBtn,2,1)
        grid.addWidget(self.restart_btn,2,2)
        grid.addWidget(self.infoLabel,3,1)
        grid.addWidget(QtGui.QLabel('Humans'),4,1)
        grid.addWidget(QtGui.QLabel('Machines'),4,2)
        grid.addWidget(self.human_label,5,1,2,1)
        grid.addWidget(self.machine_label,5,2,2,1)
        
        
        grid.addWidget(QtGui.QLabel('Your Guess'),7,1,2,1)
        grid.addWidget(QtGui.QLabel('Machine\'s Guess'),7,2,2,1)
        grid.addWidget(QtGui.QLabel('Actual Region'),7,3,2,1)        
        
        grid.addWidget(self.guess_label,8,1,2,1)
        grid.addWidget(self.pred_label,8,2,2,1)
        grid.addWidget(self.correct_label,8,3,2,1)
        
        RegionGrid = QtGui.QGridLayout()
          
        self.regionBtns = []        
        for i in range(7):
            self.regionBtns.append(QtGui.QPushButton(s.REGIONS[i],self))
        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)
        for i in range(7):
            RegionGrid.addWidget(self.regionBtns[i],0,i)
            self.regionBtns[i].setEnabled(False)
            self.regionBtns[i].clicked.connect(self.regionCallback)

        vbox = QtGui.QVBoxLayout()
        
        vbox.addLayout(title)
        vbox.addStretch(0.5)
        vbox.addLayout(grid)
        vbox.addStretch(1)
        vbox.addLayout(RegionGrid)
        
        self.setLayout(vbox)
    
        self.show()
    
        
    def center(self):
        
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    def closeEvent(self, event):
        
        reply = QtGui.QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QtGui.QMessageBox.Yes | 
            QtGui.QMessageBox.No, QtGui.QMessageBox.No)

        if reply == QtGui.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()     

def main():
    
    app = QtGui.QApplication(sys.argv)
    
    ex = Example()
    
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()