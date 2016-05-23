# -*- coding: utf-8 -*-
"""
Created on Sat May 21 09:58:56 2016

@author: James
"""
from sklearn import linear_model
from statsmodels.tsa import stattools
import numpy as np
from scipy import stats

class Classifier:
    '''
    Classifier class. Extracts features/makes a prediction based on input audio
    data.
    '''
    def __init__(self,phi):
        self.phi = phi
        self.predictor = None
    def extract_features(self,X):
        #return self.phi(X)
        return self.phi.get_phi(X)
    def make_prediction(self,X):
        phiX = self.phi.get_phi(X)
        votes = self.predictor.predict(phiX)
        return stats.mode(votes).mode[0]

    def trainLogitBatch2(self,train_samples,test_samples,C=500):
        '''
        Train a logistic regression that separates a supersample into a batch 
        of samples. The samples are then used to train a logistic regression as
        normal. However, predictions are made by breaking the input into 
        smaller samples and then using this batch of samples to determine the 
        final output label.
        
        train_samples = array of supersamples for training
        test_samples  = array of supersamples for testing
        C             = Logistic regularization parameter
        '''
        n_train = len(train_samples);
        #n_test = len(test_samples);

        samples_per = train_samples[0].N;

        X_train = np.zeros((samples_per*n_train,self.phi.LEN))
        Y_train = np.zeros(samples_per*n_train,dtype=np.int8);

        k = 0
        for super_sample in train_samples:
            phi_X = self.phi.get_phi(super_sample)
            numSamples,_ = phi_X.shape
            X_train[k:k+numSamples,:] = phi_X
            Y_train[k:k+numSamples] = super_sample.region
            k += numSamples

        log_reg = linear_model.LogisticRegression(C=C)
        log_reg.fit(X_train,Y_train)
        
        self.predictor = log_reg

        train_actual = np.zeros((n_train,1))
        train_hat = np.zeros((n_train,1))

        for i,super_sample in enumerate(train_samples):
            train_actual[i] = super_sample.region
            train_hat[i] = self.make_prediction(super_sample)

        print("Finished Training Logistic Classifier with Training Error:---------------")
        for region in range(7):
            actual = train_actual[train_actual == region]
            pred = train_hat[train_actual == region]
            err = 1 - float(sum(actual == pred))/len(actual)
            print "Error for region %d: %.4f" % (region,err)
        totalErr = 1 - float(sum(train_actual == train_hat))/len(train_actual)
        print "---- Total Training Error: %.4f" % totalErr


    def trainLogitBatch(self,train_samples,test_samples, audio_dur=60, sample_length=10, acf_lags = 40, fft_bins = 40):
        '''
        train_samples   = array of super samples to be trained on
        test_samples    = array of super samples to be tested on
        phi             = feature extractor on a supersample
        '''
        if self.predictor != None:
            raise ValueError("Error: This classifier has already trained a predictor.")
            
        samples_per = int(audio_dur/sample_length)

        # Allocate more room then we expect, just in case
        X = np.zeros((2*samples_per*len(train_samples),acf_lags+1+fft_bins+1+1))
        Y = np.zeros(2*samples_per*len(train_samples),dtype=np.int8)

        k = 0
        for super_sample in train_samples:
            phi_X = self.phi(super_sample)
            numSamples,_ = phi_X.shape
            X[k:k+numSamples,:] = phi_X
            Y[k:k+numSamples] = super_sample.region
            k += numSamples

        X_train = X[:k,:]
        Y_train = Y[:k]

        log_reg = linear_model.LogisticRegression(C=500)
        log_reg.fit(X_train,Y_train)
        
        self.predictor = log_reg

        train_actual = np.zeros((len(train_samples),1))
        train_hat = np.zeros((len(train_samples),1))

        for super_sample,i in zip(train_samples,range(len(train_samples))):
            train_actual[i] = super_sample.region
            train_hat[i] = self.make_prediction(super_sample)

        print("Finished Training Logistic Classifier with Training Error:---------------")
        for region in range(7):
            actual = train_actual[train_actual == region]
            pred = train_hat[train_actual == region]
            err = 1 - float(sum(actual == pred))/len(actual)
            print "Error for region %d: %.4f" % (region,err)
        totalErr = 1 - float(sum(train_actual == train_hat))/len(train_actual)
        print "---- Total Training Error: %.4f" % totalErr

        #TODO figure out best way to return testing data
