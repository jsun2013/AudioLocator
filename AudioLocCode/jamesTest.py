# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:26:24 2016

@author: james
"""

import numpy as np
from sklearn import linear_model
from statsmodels.tsa import stattools
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import samples
reload(samples)
import spectral
reload(spectral)
import audiolearning
reload(audiolearning)
import mytimer as mt
from scikits.talkbox import features


np.random.seed(100)

FFT_BINS = 40;
ACF_LAGS = 40

class phi1:
    LEN = 0;
    def __init__(self,acf_lags,nceps):
#        self.LEN = nceps+acf_lags+1
        self.LEN = nceps
        self.acf_lags = acf_lags

    def get_phi(self,sample):
        '''
        Takes in a super_sample and returns a feature array. Breaks the super_sample
        down into samples. Each row of the returned value corresponds to a sample
        in the super sample.
        '''

#        XSPED = spectral.getSupersampleSPED(super_sample,self.fft_bins,spacing="log")
#        XAcf = spectral.getSampleACF(sample,self.acf_lags)
#        XMean = np.zeros((sample.Nsub,1))
        XMFCC = spectral.getSampleMFCC(sample)
#        XAcf = np.zeros((sample.Nsub,self.acf_lags+1))
#        XMFCC = np.zeros((sample.Nsub,13))
#        subsamples = sample.getSubsamples()
#        for j,data in enumerate(subsamples):      
#            XMFCC[j,:] = spectral.getSignalMFCC(data)
#            XAcf[j,:] = spectral.getSignalACF(data,self.acf_lags)
#        return np.hstack((XMFCC,XMean,XAcf))
#        return np.hstack((XMFCC,XAcf))
        return XMFCC
        


class phi2:
    LEN = 0;
    def __init__(self,acf_lags,nceps):
#        self.LEN = nceps+acf_lags+1
        self.LEN = nceps + acf_lags + 1
        self.acf_lags = acf_lags

    def get_phi(self,sample):
        '''
        Takes in a super_sample and returns a feature array. Breaks the super_sample
        down into samples. Each row of the returned value corresponds to a sample
        in the super sample.
        '''

#        XSPED = spectral.getSupersampleSPED(super_sample,self.fft_bins,spacing="log")
        XAcf = spectral.getSampleACF(sample,self.acf_lags)
#        XMean = np.zeros((sample.Nsub,1))
        XMFCC = spectral.getSampleMFCC(sample)
#        XAcf = np.zeros((sample.Nsub,self.acf_lags+1))
#        XMFCC = np.zeros((sample.Nsub,13))
#        subsamples = sample.getSubsamples()
#        for j,data in enumerate(subsamples):      
#            XMFCC[j,:] = spectral.getSignalMFCC(data)
#            XAcf[j,:] = spectral.getSignalACF(data,self.acf_lags)
#        return np.hstack((XMFCC,XMean,XAcf))
        return np.hstack((XMFCC,XAcf))
        
class phi3:
    LEN = 0;
    def __init__(self,fft_bins):
#        self.LEN = nceps+acf_lags+1
        self.LEN = fft_bins
        self.fft_bins = fft_bins
#        self.acf_lags = acf_lags

    def get_phi(self,sample):
        '''
        Takes in a super_sample and returns a feature array. Breaks the super_sample
        down into samples. Each row of the returned value corresponds to a sample
        in the super sample.
        '''

#        XSPED = spectral.getSupersampleSPED(super_sample,self.fft_bins,spacing="log")
#        XAcf = spectral.getSampleACF(sample,self.acf_lags)
        XFFT = spectral.getSupersampleFFT(sample,self.fft_bins)[1]
#        XMean = np.zeros((sample.Nsub,1))
#        XMFCC = spectral.getSampleMFCC(sample)
#        XAcf = np.zeros((sample.Nsub,self.acf_lags+1))
#        XMFCC = np.zeros((sample.Nsub,13))
#        subsamples = sample.getSubsamples()
#        for j,data in enumerate(subsamples):      
#            XMFCC[j,:] = spectral.getSignalMFCC(data)
#            XAcf[j,:] = spectral.getSignalACF(data,self.acf_lags)
#        return np.hstack((XMFCC,XMean,XAcf))
        return XFFT
        

if __name__ == "__main__":
#    mt.tic()
#    
##    all_samples = samples.getAllSamples(Tsub=2,Nsub=5,READ_IN=True) #2 second subsamples, 5 per sample
##    
##    np.random.shuffle(all_samples);
##    numTrain = int(round(2*len(all_samples)/3))
##    train_samples = all_samples[:numTrain]
##    test_samples = all_samples[numTrain:]
##    
#    nfft_bins = FFT_BINS;
#    myPhi1 = phi1(ACF_LAGS,13);
#    
#    
#    
#    C_range = np.arange(5,200,10)
#    logit_test = np.zeros(C_range.shape)
#    logit_train = np.zeros(C_range.shape)
#    
#    best_C_logit = 0
#    best_err_logit = float('Inf')
#    for i,C in enumerate(C_range):        
#        logistic_classifier = audiolearning.Classifier(myPhi1);
#        
#        logit_train[i] = logistic_classifier.trainLogitBatch(train_samples=None,X_train=X_train,Y_train=Y_train,C=C);
#        logit_test[i] = logistic_classifier.testClassifier(test_samples=None,X_test=X_test,Y_test=Y_test)
#        if logit_test[i] < best_err_logit:
#            best_err_logit = logit_test[i]
#            best_C_logit = C
#        
#    svm_test = np.zeros(C_range.shape)
#    svm_train = np.zeros(C_range.shape)
#    
#    best_C_svm = 0
#    best_err_svm = float('Inf')
#    for i,C in enumerate(C_range):
#        print "SVM with C = %d" % C
#        
#        svm_classifier = audiolearning.Classifier(myPhi1)
#        svm_train[i] = svm_classifier.trainSVMBatch(train_samples=None,X_train=X_train,Y_train=Y_train,C=C)
#        svm_test[i] = svm_classifier.testClassifier(test_samples=None,X_test=X_test,Y_test=Y_test)
#        if svm_test[i] < best_err_svm:
#            best_err_svm = svm_test[i]
#            best_C_svm = C
#        
#    lin_svm_test = np.zeros(C_range.shape)
#    lin_svm_train = np.zeros(C_range.shape)
#    
#    best_C_lsvm = 0
#    best_err_lsvm = float('Inf')
#    for i,C in enumerate(C_range):
#        print "SVM with C = %d" % C
#        
#        lin_svm_classifier = audiolearning.Classifier(myPhi1)
#        lin_svm_train[i] = lin_svm_classifier.trainSVMBatch(train_samples=None,X_train=X_train,Y_train=Y_train,kernel='linear',C=C)
#        lin_svm_test[i] = lin_svm_classifier.testClassifier(test_samples=None,X_test=X_test,Y_test=Y_test)
#        if lin_svm_test[i] < best_err_svm:
#            best_err_svm = lin_svm_test[i]
#            best_C_svm = C
#    
#    sns.set_style({"legend.frameon" : True})
#    plt.plot(C_range,logit_train,'*')
#    plt.plot(C_range,logit_test) 
#    plt.plot(C_range,svm_train,'o')
#    plt.plot(C_range,svm_test)    
#    plt.plot(C_range,lin_svm_train,'--')
#    plt.plot(C_range,lin_svm_test)  
#    plt.title('MFCC')
#    plt.legend(['Logit Train','Logit Test','SVM RBF Train', 'SVM RBF Test', 'SVM Lin Train', 'SVM Lin Test'])
#    plt.xlabel('C')
#    plt.ylabel('Error')
#    
#    mt.toc()

    mt.tic()
    
    all_samples = samples.getAllSamples(Tsub=2,Nsub=5,READ_IN=True) #2 second subsamples, 5 per sample
    
    np.random.shuffle(all_samples);
    numTrain = int(round(2*len(all_samples)/3))
    train_samples = all_samples[:numTrain]
    test_samples = all_samples[numTrain:]
    
    nfft_bins = FFT_BINS;
    myPhi2 = phi1(ACF_LAGS,13);
    extractor = audiolearning.Classifier(myPhi2)
    (X_train,Y_train,num_sub_train) = extractor.extract_features(train_samples)   
    (X_test,Y_test,num_sub_test) = extractor.extract_features(test_samples)
    
    
    
    
    C_range = np.arange(100,1000,100)
    logit_test = np.zeros(C_range.shape)
    logit_train = np.zeros(C_range.shape)
    
    best_C_logit = 0
    best_err_logit = float('Inf')
    for i,C in enumerate(C_range):        
        logistic_classifier = audiolearning.Classifier(myPhi2);  
        
        logit_train[i] = logistic_classifier.trainLogitBatch(train_samples=None,X_train=X_train,Y_train=Y_train,num_subs=num_sub_train,C=C);
        logit_test[i] = logistic_classifier.testClassifier(test_samples=None,X_test=X_test,Y_test=Y_test,num_subs=num_sub_test)
        if logit_test[i] < best_err_logit:
            best_err_logit = logit_test[i]
            best_C_logit = C
        
    svm_test = np.zeros(C_range.shape)
    svm_train = np.zeros(C_range.shape)
    
    best_C_svm = 0
    best_err_svm = float('Inf')
    for i,C in enumerate(C_range):
        print "SVM with C = %d" % C
        
        svm_classifier = audiolearning.Classifier(myPhi2)
        svm_train[i] = svm_classifier.trainSVMBatch(train_samples=None,X_train=X_train,Y_train=Y_train,num_subs=num_sub_train,C=C)
        svm_test[i] = svm_classifier.testClassifier(test_samples=None,X_test=X_test,Y_test=Y_test,num_subs=num_sub_test)
        if svm_test[i] < best_err_svm:
            best_err_svm = svm_test[i]
            best_C_svm = C
        
    lin_svm_test = np.zeros(C_range.shape)
    lin_svm_train = np.zeros(C_range.shape)
    
    best_C_lsvm = 0
    best_err_lsvm = float('Inf')
    for i,C in enumerate(C_range):
        print "SVM with C = %d" % C
        
        lin_svm_classifier = audiolearning.Classifier(myPhi2)
        lin_svm_train[i] = lin_svm_classifier.trainSVMBatch(train_samples=None,X_train=X_train,Y_train=Y_train,kernel='linear',num_subs=num_sub_train,C=C)
        lin_svm_test[i] = lin_svm_classifier.testClassifier(test_samples=None,X_test=X_test,Y_test=Y_test,num_subs=num_sub_test)
        if lin_svm_test[i] < best_err_svm:
            best_err_lsvm = lin_svm_test[i]
            best_C_svm = C
    
    sns.set_style({"legend.frameon" : True})
    plt.plot(C_range,logit_train,'*')
    plt.plot(C_range,logit_test) 
    plt.plot(C_range,svm_train,'o')
    plt.plot(C_range,svm_test)    
    plt.plot(C_range,lin_svm_train,'--')
    plt.plot(C_range,lin_svm_test)  
    plt.title('ACF')
    plt.legend(['Logit Train','Logit Test','SVM RBF Train', 'SVM RBF Test', 'SVM Lin Train', 'SVM Lin Test'])
    plt.xlabel('C')
    plt.ylabel('Error')
    
    mt.toc()
