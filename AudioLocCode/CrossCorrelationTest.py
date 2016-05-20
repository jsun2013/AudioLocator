# -*- coding: utf-8 -*-
"""
Created on Fri May 20 12:06:50 2016

@author: James
"""

import samples
import numpy as np
from sklearn import linear_model
from statsmodels.tsa import stattools

np.random.seed(102304)

sampleLength = 10 # Seconds
samplesPer = int(60/sampleLength) # Num Samples per super sample
acfLags = 40;

train_samples = samples.getAllSamples()

# Allocate more room then we expect, just in case
XAcf = np.zeros((10*samplesPer*len(train_samples),acfLags+1))
XCorr = np.zeros((10*samplesPer*len(train_samples),1))
XMean = np.zeros((10*samplesPer*len(train_samples),1))
XVar = np.zeros((10*samplesPer*len(train_samples),1))
Y = np.zeros((10*samplesPer*len(train_samples),1))

'''Build autocorrelation as features'''
i = 0
for sample in train_samples:
    sample.read_parms()
    T =  int(sample.waveparms.L/sample.waveparms.fs)
    numSubSamples = int(round(T/10))
    for t in range(numSubSamples):
        Y[i] = sample.region
        data = sample.read_data(t*sampleLength,sampleLength,'sec')
        
        XCorr[i] = np.correlate(data,data,'valid')
        XAcf[i,:] = stattools.acf(data,nlags=acfLags)
        XMean[i] = np.mean(data)
        XVar[i] = np.var(data);
        i+=1
        
X = np.hstack((XAcf[:i,:],XMean[:i,:],XVar[:i,:],XCorr[:i,:]))
Y = Y[:i]

DataSet = np.hstack((X,Y))
np.random.shuffle(DataSet);

numTrain = int(round(2*len(Y)/3))

XTrain = DataSet[:numTrain,:-1]
XTest = DataSet[numTrain:,:-1]


YTrain = DataSet[:numTrain,-1]
YTest = DataSet[numTrain:,-1]


log_reg = linear_model.LogisticRegression(C=100)


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
print "---- Total Training Error: %.4f" % totalErr