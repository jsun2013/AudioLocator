# -*- coding: utf-8 -*-
"""
Created on Fri May 20 12:06:50 2016

@author: James
"""

import samples
import numpy as np
from sklearn import linear_model
from statsmodels.tsa import stattools
import spectral

np.random.seed(25)

#sampleLength = 10 # Seconds
#samplesPer = int(60/sampleLength) # Num Samples per super sample
#acfLags = 40
#FFTBins = 40
#train_samples = samples.getAllSamples(T=sampleLength,N=samplesPer)
#
## Allocate more room then we expect, just in case
#XAcf = np.zeros((10*samplesPer*len(train_samples),acfLags+1))
#XFFT = np.zeros((10*samplesPer*len(train_samples),FFTBins))
#XCorr = np.zeros((10*samplesPer*len(train_samples),1))
#XMean = np.zeros((10*samplesPer*len(train_samples),1))
#XVar = np.zeros((10*samplesPer*len(train_samples),1))
#Y = np.zeros((10*samplesPer*len(train_samples),1))
#
#
#'''Build autocorrelation as features'''
#F_all = spectral.getAllFFT(train_samples, FFTBins)
#k = 0
#for superSample,i in zip(train_samples,range(len(train_samples))):
#    for data,j in zip(superSample.samples,range(len(superSample.samples))):
#        Y[k] = superSample.region   
#        XFFT[k,:] = F_all[i,j,:]
#        XAcf[k,:] = stattools.acf(data,nlags=acfLags,fft=True)
#        XMean[k] = np.mean(data)
#        XVar[k] = np.var(data);
#        k+=1

#X = np.hstack((XAcf[:k,:],XMean[:k,:],XVar[:k,:]))      
X = np.hstack((XAcf[:k,:],XFFT[:k,:],XMean[:k,:],XVar[:k,:]))
Y = Y[:k]

DataSet = np.hstack((X,Y))
np.random.shuffle(DataSet);

numTrain = int(round(2*len(Y)/3))

XTrain = DataSet[:numTrain,:-1]
XTest = DataSet[numTrain:,:-1]


YTrain = DataSet[:numTrain,-1]
YTest = DataSet[numTrain:,-1]

print("----------------Logistic Regression------------------")
print("-----------------------------------------------------")
print("-----------------------------------------------------")
print("-----------------------------------------------------")
log_reg = linear_model.LogisticRegression(C=500)
log_reg.fit(XTrain,YTrain)

Yhat_test = log_reg.predict(XTest)
Yhat_train = log_reg.predict(XTrain)

print("-------------------Training Error:-------------------")
for region in range(7):
    actual = YTrain[YTrain == region]
    pred = Yhat_train[YTrain == region]
    err = 1 - float(sum(actual == pred))/len(actual)
    print "Error for region %d: %.4f" % (region,err)
totalErr = 1 - float(sum(YTrain == Yhat_train))/len(YTrain)
print "---- Total Training Error: %.4f" % totalErr


print("-----------------------------------------------------")
print("-----------------------------------------------------")
print("-----------------------------------------------------")
print("-------------------Testing Error:-------------------")
for region in range(7):
    actual = YTest[YTest == region]
    pred = Yhat_test[YTest == region]
    err = 1 - float(sum(actual == pred))/len(actual)
    print "Error for region %d: %.4f" % (region,err)
totalErr = 1 - float(sum(YTest == Yhat_test))/len(YTest)
print "---- Total Testing Error: %.4f" % totalErr


print("----------------------- KNN -------------------------")
print("-----------------------------------------------------")
print("-----------------------------------------------------")
print("-----------------------------------------------------")
from sklearn.neighbors import KNeighborsClassifier

for n in range(2,8):
    print ""
    print "****N = %d" %n
    neigh = KNeighborsClassifier(n_neighbors=n)
    neigh.fit(XTrain,YTrain)
    
    
    Yhat_test = neigh.predict(XTest)
    Yhat_train = neigh.predict(XTrain)
    
    
    print("-------------------Training Error:-------------------")
    for region in range(7):
        actual = YTrain[YTrain == region]
        pred = Yhat_train[YTrain == region]
        err = 1 - float(sum(actual == pred))/len(actual)
        print "Error for region %d: %.4f" % (region,err)
    totalErr = 1 - float(sum(YTrain == Yhat_train))/len(YTrain)
    print "---- Total Training Error: %.4f" % totalErr
    
    
    print("-----------------------------------------------------")
    print("-----------------------------------------------------")
    print("-----------------------------------------------------")
    print("-------------------Testing Error:-------------------")
    for region in range(7):
        actual = YTest[YTest == region]
        pred = Yhat_test[YTest == region]
        err = 1 - float(sum(actual == pred))/len(actual)
        print "Error for region %d: %.4f" % (region,err)
    totalErr = 1 - float(sum(YTest == Yhat_test))/len(YTest)
    print "---- Total Testing Error: %.4f" % totalErr
    
print ""
print("----------------------- SVM -------------------------")
print("-----------------------------------------------------")
print("-----------------------------------------------------")
print("-----------------------------------------------------")
from sklearn import svm
clf = svm.SVC(C=100)
clf.fit(XTrain,YTrain)

Yhat_test = clf.predict(XTest)
Yhat_train = clf.predict(XTrain)

print("-------------------Training Error:-------------------")
for region in range(7):
    actual = YTrain[YTrain == region]
    pred = Yhat_train[YTrain == region]
    err = 1 - float(sum(actual == pred))/len(actual)
    print "Error for region %d: %.4f" % (region,err)
totalErr = 1 - float(sum(YTrain == Yhat_train))/len(YTrain)
print "---- Total Training Error: %.4f" % totalErr


print("-----------------------------------------------------")
print("-----------------------------------------------------")
print("-----------------------------------------------------")
print("-------------------Testing Error:-------------------")
for region in range(7):
    actual = YTest[YTest == region]
    pred = Yhat_test[YTest == region]
    err = 1 - float(sum(actual == pred))/len(actual)
    print "Error for region %d: %.4f" % (region,err)
totalErr = 1 - float(sum(YTest == Yhat_test))/len(YTest)
print "---- Total Testing Error: %.4f" % totalErr
