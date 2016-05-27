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

    self.phi is a function that creates a feature matrix for a sample
    '''
    def __init__(self,phi):
        self.phi = phi
        self.predictor = None
    def isTrained(self):
        return self.predictor == None
    def extract_features(self,samples):
        #k = 0
        n_train = len(samples);
        #n_test = len(test_samples);

        subsamp_per = samples[0].Nsub;

        X = np.zeros((n_train,subsamp_per,self.phi.LEN));
        #Y= np.zeros(n_train,subsamp_per,dtype=np.int8);
        Y= np.zeros(n_train,dtype=np.int8);

        #X = np.zeros((subsamp_per*n_train,self.phi.LEN))
        #Y= np.zeros(subsamp_per*n_train,dtype=np.int8);
        #num_subs = np.zeros(n_train,dtype=np.int8)
        print("Running feature extraction...")
        nupdate = int(n_train/10);
        for i,sample in enumerate(samples):
            phi_X = self.phi.get_phi(sample)
            #numSamples,_ = phi_X.shape
            X[i,:,:] = phi_X
            Y[i] = sample.region
            if i%nupdate==0:
                print("%d%%..."%((100*i)/n_train));
            #num_subs[i] = numSamples
            #k += numSamples
        #return (X,Y,num_subs)
        return (X,Y)
    def make_prediction(self,X):
        phiX = self.phi.get_phi(X)
        votes = self.predictor.predict(phiX)
        return stats.mode(votes).mode[0]
    def make_batch_prediction(self,phi_x,num_subs):
        '''
        phi_x is a matrix of feature vectors of subsamples
        num_subs is a vector of counts for each sample. Thus, if num_subs[0] = 4,
            then phi_x[0:4,:] corresponds to sample 1
        Typical usage:
            (X,Y,num_subs) = extractor.extract_features(s)
            m = len(num_subs)
            hat = self.make_batch_prediction(X,num_subs)
            actual = np.zeros(m)
            k = 0
            for i,nsub in enumerate(num_subs):
                actual[i] = Y_test[k];
                k+=nsub
        '''
        m, nsub, nfeat = np.shape(phi_x);
        hat = np.zeros(m);
        sub_hat = self.predictor.predict(np.reshape(phi_x,(m*nsub,nfeat)));
        for i in range(m):
            votes = sub_hat[i*nsub:(i+1)*nsub]
            hat[i] = stats.mode(votes).mode[0]
        """
        k = 0
        hat = np.zeros(len(num_subs))
        sub_hat = self.predictor.predict(phi_x)
        for i,nsub in enumerate(num_subs):
            votes = sub_hat[k:k+nsub]
            hat[i] = stats.mode(votes).mode[0]
            k += nsub
        """
        return hat

    def make_phi_prediction(self,phi_x):
        return self.predictor.predict(phi_x)
    def trainSVMBatch(self,train_samples,X_train=None,Y_train=None,num_subs=None,kernel='rbf',C=500,gamma='auto'):
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

        #if X_train is None or Y_train is None or num_subs is None:
        if X_train is None or Y_train is None:
            n_train = len(train_samples);

            subsamp_per = train_samples[0].Nsub;

            X_train = np.zeros((subsamp_per*n_train,self.phi.LEN))
            Y_train = np.zeros(subsamp_per*n_train,dtype=np.int8);

            k = 0
            print("Running feature extraction...")
            nupdate = int(n_train/10);
            for i,sample in enumerate(train_samples):
                if i%nupdate==0:
                    print("%d%%..."%((100*i)/n_train));
                phi_X = self.phi.get_phi(sample)
                numSamples,_ = phi_X.shape
                X_train[k:k+numSamples,:] = phi_X
                Y_train[k:k+numSamples] = sample.region
                k += numSamples
            print("Finished feature extraction. Running fitting...");
            if kernel=='rbf':
                clf = svm.SVC(kernel=kernel,C=C,gamma=gamma)
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
        else:
            m, nsub, nfeat = np.shape(X_train);
            #n_train = len(num_subs)

            if kernel=='rbf':
                clf = svm.SVC(kernel=kernel,C=C,gamma=gamma)
            elif kernel=='linear':
                clf = svm.LinearSVC(C=C,loss='hinge')
            #clf.fit(X_train,Y_train)
            clf.fit(np.reshape(X_train,(m*nsub,nfeat)),np.repeat(Y_train,nsub));

            self.predictor = clf

            #train_actual = np.zeros(n_train)
            train_actual = Y_train;
            #train_hat = self.make_batch_prediction(X_train,num_subs)
            train_hat = self.make_batch_prediction(X_train,None);
            """
            k = 0
            for i,nsub in enumerate(num_subs):
                train_actual[i] = Y_train[k];
                k+=nsub
            """

            print("Finished Training Classifier with Training Error:---------------")
            for region in range(7):
                actual = train_actual[train_actual == region]
                pred = train_hat[train_actual == region]
                err = 1 - float(sum(actual == pred))/len(actual)
                print "Error for region %d: %.4f" % (region,err)
            totalErr = 1 - float(sum(train_actual == train_hat))/len(train_actual)
            print "---- Total Training Error: %.4f" % totalErr
        return totalErr

    def trainLogitBatch2(self,train_samples,test_samples=None,C=500):
        '''
        Deprecated, does  the same as trainLogitBatch
        '''
        warnings.warn("trainLogitBatch2 does the same as trainLogitBatch. Switch to using that one")
        return self.trainLogitBatch(train_samples,C)


    def trainLogitBatch(self,train_samples,X_train=None,Y_train=None,num_subs=None,C=500):
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

        if X_train is None or Y_train is None or num_subs is None:
            n_train = len(train_samples);

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
        else:
            n_train = len(num_subs)
            log_reg = linear_model.LogisticRegression(C=C)
            log_reg.fit(X_train,Y_train)

            self.predictor = log_reg

            train_actual = np.zeros(n_train)
            train_hat = self.make_batch_prediction(X_train,num_subs)
            k = 0
            for i,nsub in enumerate(num_subs):
                train_actual[i] = Y_train[k];
                k+=nsub


            print("Finished Training Classifier with Training Error:---------------")
            for region in range(7):
                actual = train_actual[train_actual == region]
                pred = train_hat[train_actual == region]
                err = 1 - float(sum(actual == pred))/len(actual)
                print "Error for region %d: %.4f" % (region,err)
            totalErr = 1 - float(sum(train_actual == train_hat))/len(train_actual)
            print "---- Total Training Error: %.4f" % totalErr
        return totalErr


    def testClassifier(self, test_samples,X_test=None,Y_test=None,num_subs=None):
        if self.predictor == None:
            raise ValueError("Error: This classifier has not been trained yet.")
        #if X_test is None or Y_test is None or num_subs is None:
        if X_test is None or Y_test is None:
            n_test = len(test_samples);
            test_actual = np.zeros((n_test,1))
            test_hat = np.zeros((n_test,1))
            for (i,sample) in enumerate(test_samples):
                test_actual[i] = sample.region
                test_hat[i] = self.make_prediction(sample)
        else:
            n_test, nsub, nfeat = np.shape(X_test);
            #n_test = len(num_subs)
            test_actual = np.zeros(n_test)
            #test_hat = self.make_batch_prediction(X_test,num_subs)
            test_hat = self.make_batch_prediction(X_test,None)
            test_actual=Y_test;
            """
            k = 0
            for i,nsub in enumerate(num_subs):
                test_actual[i] = Y_test[k];
                k+=nsub
            """


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

        return totalErr

