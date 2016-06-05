# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 09:41:06 2016

@author: ReidW
"""
import os
import csv

csv_file = 'testTracker.csv'
if not os.path.exists(csv_file):
    with open(csv_file,'wb') as mycsv:
        writer = csv.writer(mycsv);
        writer.writerow(['User','Classifier','Actual'])

def addEntry(user, classifier, actual):
    with open(csv_file,'ab') as mycsv:
        writer = csv.writer(mycsv);
        writer.writerow([user, classifier, actual]);

def getStats():
    user_err=0;
    class_err=0;
    total=0;
    with open(csv_file,'rb') as mycsv:
        reader = csv.reader(mycsv);
        for row in reader:
            if row[0] == "User":
                pass
            user = row[0];
            classifier = row[1];
            actual = row[2];
            if user!=actual:
                user_err += 1
            if classifier != actual:
                class_err+=1
            total+=1
    print "User error rate = %0.02f%%"%(100.0*float(user_err)/total);
    print "Classifier error rate = %0.02f%%"%(100.0*float(class_err)/total);
    user_correct= total-user_err
    class_correct = total-class_err
    return (user_correct,class_correct)