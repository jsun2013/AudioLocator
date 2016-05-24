# -*- coding: utf-8 -*-
"""
Created on Sat May 21 09:58:56 2016

@author: James
"""
from sklearn import linear_model,svm,ensemble
from statsmodels.tsa import stattools
import numpy as np
from scipy import stats
import warnings

class Classifier:
    '''
    Classifier class. Extracts features/makes a prediction based on input audio
    data.
    '''
    def __init__(self,phi):
        self.phi = phi
        self.predictor = None
    def isTrained(self):
        return self.predictor == None
    def extract_features(self,X):
        #return self.phi(X)
        return self.phi.get_phi(X)
    def make_prediction(self,X):
        phiX = self.phi.get_phi(X)
        votes = self.predictor.predict(phiX)
        return stats.mode(votes).mode[0]

    def trainSVMBatch(self,train_samples,kernel='rbf',C=500,gamma='auto'):
        '''
        Train a kernalized SVM that separates a supersample into a batch
        of samples. The samples are then used to train the SVM as
        normal. However, predictions are made by breaking the input into
        smaller samples and then using this batch of samples to determine the
        final output label.

        train_samples = array of supersamples for training
        test_samples  = array of supersamples for testing
        kernel        = Kernel function to use
        C             = Logistic regularization parameter
        '''
        n_train = len(train_samples);
        #n_test = len(test_samples);

        subsamp_per = train_samples[0].Nsub;

        X_train = np.zeros((subsamp_per*n_train,self.phi.LEN))
        Y_train = np.zeros(subsamp_per*n_train,dtype=np.int8);

        k = 0
        print("Running feature extraction...")
        nupdate = int(n_train/10);
        for (i,sample) in enumerate(train_samples):
            if i%nupdate==0:
                print("%d%%..."%((100*i)/n_train));
            phi_X = self.phi.get_phi(sample)
            numSamples,_ = phi_X.shape
            X_train[k:k+numSamples,:] = phi_X
            Y_train[k:k+numSamples] = sample.region
            k += numSamples

        print("Finished feature extraction. Running fitting...");
        if kernel=='rbf':
            clf = svm.SVC(C=C,gamma=gamma)
        elif kernel=='linear':
            clf = svm.LinearSVC(C=C,loss='hinge')

        clf.fit(X_train,Y_train)

        self.predictor = clf

        train_actual = np.zeros((n_train,1))
        train_hat = np.zeros((n_train,1))
        print("Making predictions...");

        for i,sample in enumerate(train_samples):
            train_actual[i] = sample.region
            train_hat[i] = self.make_prediction(sample)

        print("Finished Training Classifier with Training Error:---------------")
        for region in range(7):
            actual = train_actual[train_actual == region]
            pred = train_hat[train_actual == region]
            err = 1 - float(sum(actual == pred))/len(actual)
            print "Error for region %d: %.4f" % (region,err)
        totalErr = 1 - float(sum(train_actual == train_hat))/len(train_actual)
        print "---- Total Training Error: %.4f" % totalErr

    def trainLogitBatch2(self,train_samples,test_samples=None,C=500):
        '''
        Deprecated, does  the same as trainLogitBatch
        '''
        warnings.warn("trainLogitBatch2 does the same as trainLogitBatch. Switch to using that one")
        self.trainLogitBatch(train_samples,C)


    def trainLogitBatch(self,train_samples,C=500):
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

        subsamp_per = train_samples[0].Nsub;

        X_train = np.zeros((subsamp_per*n_train,self.phi.LEN))
        Y_train = np.zeros(subsamp_per*n_train,dtype=np.int8);

        k = 0
        for sample in train_samples:
            phi_X = self.phi.get_phi(sample)
            numSamples,_ = phi_X.shape
            X_train[k:k+numSamples,:] = phi_X
            Y_train[k:k+numSamples] = sample.region
            k += numSamples

        log_reg = linear_model.LogisticRegression(C=C)
        log_reg.fit(X_train,Y_train)

        self.predictor = log_reg

        train_actual = np.zeros((n_train,1))
        train_hat = np.zeros((n_train,1))

        for i,sample in enumerate(train_samples):
            train_actual[i] = sample.region
            train_hat[i] = self.make_prediction(sample)

        print("Finished Training Classifier with Training Error:---------------")
        for region in range(7):
            actual = train_actual[train_actual == region]
            pred = train_hat[train_actual == region]
            err = 1 - float(sum(actual == pred))/len(actual)
            print "Error for region %d: %.4f" % (region,err)
        totalErr = 1 - float(sum(train_actual == train_hat))/len(train_actual)
        print "---- Total Training Error: %.4f" % totalErr


    def testClassifier(self, test_samples):
        if self.predictor == None:
            raise ValueError("Error: This classifier has not been trained yet.")
        n_test = len(test_samples);
        test_actual = np.zeros((n_test,1))
        test_hat = np.zeros((n_test,1))
        for (i,sample) in enumerate(test_samples):
            test_actual[i] = sample.region
            test_hat[i] = self.make_prediction(sample)

        print("-----------------------------------------------------")
        print("-------------------Testing Error:-------------------")
        for region in range(7):
            actual = test_actual[test_actual == region]
            pred = test_hat[test_actual == region]
            if len(actual)==0:
                print("  ->No test samples from Region %d"%region)
                continue
            err = 1 - float(sum(actual == pred))/len(actual)
            print "Error for region %d: %.4f" % (region,err)
        totalErr = 1 - float(sum(test_actual == test_hat))/n_test
        print "---- Total Testing Error: %.4f" % totalErr

