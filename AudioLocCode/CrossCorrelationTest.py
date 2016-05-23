# -*- coding: utf-8 -*-
"""
Created on Fri May 20 12:06:50 2016

@author: James
"""

import samples
reload(samples)
import numpy as np
from sklearn import linear_model
from statsmodels.tsa import stattools
import spectral
reload(spectral)
import audiolearning
reload(audiolearning)
from scipy import stats

np.random.seed(25)

def phi(super_sample, audio_dur=60, sample_length=10, acf_lags = 40, fft_bins = 40):
    '''
    Takes in a super_sample and returns a feature array. Breaks the super_sample 
    down into samples. Each row of the returned value corresponds to a sample
    in the super sample.
    '''    
    
    samples_per = int(audio_dur/sample_length)
    XAcf = np.zeros((2*samples_per,acf_lags+1))
#    XFFT = np.zeros((2*samples_per,fft_bins))
    XMean = np.zeros((2*samples_per,1))
    XVar = np.zeros((2*samples_per,1))
    
    _,F_all = spectral.getSupersampleFFT(super_sample,fft_bins)
    for data,j in zip(super_sample.samples,range(len(super_sample.samples))):
#        XFFT[j,:] = F_all[j,:]
        XAcf[j,:] = stattools.acf(data,nlags=acf_lags,fft=True)
        XMean[j] = np.mean(data)
        XVar[j] = np.var(data);
#    return np.hstack((XAcf[:j+1,:],XFFT[:j+1,:],XMean[:j+1,:],XVar[:j+1,:]))
    return np.hstack((XAcf[:j+1,:],XMean[:j+1,:],XVar[:j+1,:]))

audio_dur = 60 # Seconds
sample_length = 10 # Seconds
samples_per = int(audio_dur/sample_length) # Num Samples per super sample
acf_lags = 40
fft_bins = 40
all_samples = samples.getAllSamples(T=sample_length,N=samples_per)

np.random.shuffle(all_samples);
numTrain = int(round(2*len(all_samples)/3))
train_samples = all_samples[:numTrain]
test_samples = all_samples[numTrain:]
logistic_classifier = audiolearning.Classifier(phi)
logistic_classifier.trainLogitBatch(train_samples,test_samples)

#logistic_classifier = audiolearning.trainLogitBatch(train_samples,test_samples,phi)

## Allocate more room then we expect, just in case
#X = np.zeros((2*samples_per*len(train_samples),acf_lags+1+fft_bins+1+1))
#Y = np.zeros(2*samples_per*len(train_samples),dtype=np.int8)
#
##XAcf = np.zeros((10*samples_per*len(train_samples),acf_lags+1))
##XFFT = np.zeros((10*samples_per*len(train_samples),fft_bins))
##XCorr = np.zeros((10*samples_per*len(train_samples),1))
##XMean = np.zeros((10*samples_per*len(train_samples),1))
##XVar = np.zeros((10*samples_per*len(train_samples),1))
##Y = np.zeros((10*samples_per*len(train_samples),1))
#
#
#'''Build features'''
#k = 0
#for super_sample in train_samples:    
#    phi_X = phi(super_sample)
#    numSamples,_ = phi_X.shape
#    X[k:k+numSamples,:] = phi_X
#    Y[k:k+numSamples] = super_sample.region
#    k += numSamples
#
#X_train = X[:k,:]
#Y_train = Y[:k]
#
##DataSet = np.hstack((X,Y))
##np.random.shuffle(DataSet);
#
##X_train = DataSet[:numTrain,:-1]
##XTest = DataSet[numTrain:,:-1]
##
##
##Y_train = DataSet[:numTrain,-1]
##YTest = DataSet[numTrain:,-1]
#
#print("----------------Logistic Regression------------------")
#print("-----------------------------------------------------")
#print("-----------------------------------------------------")
#print("-----------------------------------------------------")
#log_reg = linear_model.LogisticRegression(C=500)
#log_reg.fit(X_train,Y_train)
#def logistic_predictor(X):
#    votes = log_reg.predict(X)
#    return stats.mode(votes).mode[0]
#
#logistic_classifier = audiolearning.Classifier(logistic_predictor,phi)
#
#train_actual = np.zeros((len(train_samples),1))
#train_hat = np.zeros((len(train_samples),1))
#
#test_actual = np.zeros((len(test_samples),1))
#test_hat = np.zeros((len(test_samples),1))
#for super_sample,i in zip(train_samples,range(len(train_samples))):
#    train_actual[i] = super_sample.region
#    train_hat[i] = logistic_classifier.make_prediction(super_sample)
#
#print("-------------------Training Error:-------------------")
#for region in range(7):
#    actual = train_actual[train_actual == region]
#    pred = train_hat[train_actual == region]
#    err = 1 - float(sum(actual == pred))/len(actual)
#    print "Error for region %d: %.4f" % (region,err)
#totalErr = 1 - float(sum(train_actual == train_hat))/len(train_actual)
#print "---- Total Training Error: %.4f" % totalErr


#Yhat_test = log_reg.predict(XTest)
#Yhat_train = log_reg.predict(X_train)

#print("-------------------Training Error:-------------------")
#for region in range(7):
#    actual = Y_train[Y_train == region]
#    pred = Yhat_train[Y_train == region]
#    err = 1 - float(sum(actual == pred))/len(actual)
#    print "Error for region %d: %.4f" % (region,err)
#totalErr = 1 - float(sum(Y_train == Yhat_train))/len(Y_train)
#print "---- Total Training Error: %.4f" % totalErr


#print("-----------------------------------------------------")
#print("-----------------------------------------------------")
#print("-----------------------------------------------------")
#print("-------------------Testing Error:-------------------")
#for super_sample,i in zip(test_samples,range(len(test_samples))):
#    test_actual[i] = super_sample.region
#    test_hat[i] = logistic_classifier.make_prediction(super_sample)
#for region in range(7):
#    actual = test_actual[test_actual == region]
#    pred = test_hat[test_actual == region]
#    err = 1 - float(sum(actual == pred))/len(actual)
#    print "Error for region %d: %.4f" % (region,err)
#totalErr = 1 - float(sum(train_actual == train_hat))/len(train_actual)
#print "---- Total Testing Error: %.4f" % totalErr
#
#
#print("----------------------- KNN -------------------------")
#print("-----------------------------------------------------")
#print("-----------------------------------------------------")
#print("-----------------------------------------------------")
#from sklearn.neighbors import KNeighborsClassifier
#
#for n in range(2,8):
#    print ""
#    print "****N = %d" %n
#    neigh = KNeighborsClassifier(n_neighbors=n)
#    neigh.fit(X_train,Y_train)
#    
#    
#    Yhat_test = neigh.predict(XTest)
#    Yhat_train = neigh.predict(X_train)
#    
#    
#    print("-------------------Training Error:-------------------")
#    for region in range(7):
#        actual = Y_train[Y_train == region]
#        pred = Yhat_train[Y_train == region]
#        err = 1 - float(sum(actual == pred))/len(actual)
#        print "Error for region %d: %.4f" % (region,err)
#    totalErr = 1 - float(sum(Y_train == Yhat_train))/len(Y_train)
#    print "---- Total Training Error: %.4f" % totalErr
#    
#    
#    print("-----------------------------------------------------")
#    print("-----------------------------------------------------")
#    print("-----------------------------------------------------")
#    print("-------------------Testing Error:-------------------")
#    for region in range(7):
#        actual = YTest[YTest == region]
#        pred = Yhat_test[YTest == region]
#        err = 1 - float(sum(actual == pred))/len(actual)
#        print "Error for region %d: %.4f" % (region,err)
#    totalErr = 1 - float(sum(YTest == Yhat_test))/len(YTest)
#    print "---- Total Testing Error: %.4f" % totalErr
#    
#print ""
#print("----------------------- SVM -------------------------")
#print("-----------------------------------------------------")
#print("-----------------------------------------------------")
#print("-----------------------------------------------------")
#from sklearn import svm
#clf = svm.SVC(C=100)
#clf.fit(X_train,Y_train)
#
#Yhat_test = clf.predict(XTest)
#Yhat_train = clf.predict(X_train)
#
#print("-------------------Training Error:-------------------")
#for region in range(7):
#    actual = Y_train[Y_train == region]
#    pred = Yhat_train[Y_train == region]
#    err = 1 - float(sum(actual == pred))/len(actual)
#    print "Error for region %d: %.4f" % (region,err)
#totalErr = 1 - float(sum(Y_train == Yhat_train))/len(Y_train)
#print "---- Total Training Error: %.4f" % totalErr
#
#
#print("-----------------------------------------------------")
#print("-----------------------------------------------------")
#print("-----------------------------------------------------")
#print("-------------------Testing Error:-------------------")
#for region in range(7):
#    actual = YTest[YTest == region]
#    pred = Yhat_test[YTest == region]
#    err = 1 - float(sum(actual == pred))/len(actual)
#    print "Error for region %d: %.4f" % (region,err)
#totalErr = 1 - float(sum(YTest == Yhat_test))/len(YTest)
#print "---- Total Testing Error: %.4f" % totalErr
