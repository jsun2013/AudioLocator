# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:45:34 2016

@author: ReidW
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

import samples
reload(samples)
import spectral
reload(spectral)
import audiolearning
reload(audiolearning)
import mytimer as mt

#PARAMETERS
FFT_BINS = 60;
FWIN = 25;
TWIN = 1;
NPERSEG = 1024;
TSUB = 2;
NSUB = 1;

#Use standard filename to make sure you get the parameters you want
data_file = 'spedPhi_%i_%i_%i_%i_%i_%i.pkl'%(TSUB,NSUB,FFT_BINS,FWIN,TWIN,NPERSEG);
LOAD_DATA = True; #If possible, reuse existing Phi matrix

basis_file = 'pcaBasis_%i_%i_%i_%i.pkl'%(FFT_BINS,FWIN,TWIN,NPERSEG)

class phi1:
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

        XSPED = spectral.getSupersampleSPED(sample,self.fft_bins,fwin=FWIN,twin = TWIN,nperseg = NPERSEG,spacing="log")
        XMFCC = spectral.getSampleMFCC(sample)
        return np.hstack((XMFCC,XSPED))

class pcaBasis:
    eigvals = [];
    eigvecs = [];

def generate_basis():
    #This is only called if the basis hasn't already been found based on all data

    #Create a classifier instance with your feature extraction method
    myPhi = phi1(FFT_BINS);
    extractor = audiolearning.Classifier(myPhi);

    #Extract all features once
    if LOAD_DATA and os.path.exists(data_file):
        with open(data_file,"rb") as myPkl:
            thisData = pickle.load(myPkl);
        X = thisData.X;
        Y = thisData.Y;
        nsamp, _, _ = np.shape(X);
    else:
        #Collect all samples, figure out how many are reserved for test in X-validation
        all_samples = samples.getAllSamples(Tsub=TSUB,Nsub=NSUB,READ_IN=False) #
        nsamp = len(all_samples);

        (X,Y) = extractor.extract_features(all_samples)
        thisData = samples.DataPhi();
        thisData.X=X;
        thisData.Y=Y;
        with open(data_file,"wb") as myPkl:
            pickle.dump(thisData,myPkl);

    nsamp, nsub, nfeat = np.shape(X);
    #Flatten all subsamples out
    X = np.reshape(X,(nsamp*nsub,nfeat));
    #Normalize data
    mu = np.mean(X,0);
    Xp = X - mu;
    sigma_sq = np.mean( np.power(Xp,2) , 0);
    Xpp = np.divide(Xp,(sigma_sq+0.1)); #Smooth to prevent divide-by-zero

    #Empirical covariance
    Sig = (np.dot(Xpp.T,Xpp))/nsamp;
    (lam,U) = np.linalg.eig(Sig);

    #Sort lambda decreasing, and e-vecs accordingly
    sort_inds = np.argsort(lam);
    sort_inds = sort_inds[::-1]
    Us = U[:,sort_inds];
    lams = lam[sort_inds];

    #Stor basis file for future runs
    newBasis = pcaBasis()
    newBasis.eigvals = lams;
    newBasis.eigvecs = Us;
    with open(basis_file,'wb') as myPkl:
        pickle.dump(newBasis,myPkl);

if __name__ == "__main__":
    mt.tic()
    NDIM = 3;

    #Look for existing basis (eigenvector)
    if not os.path.exists(basis_file):
        generate_basis()

    #Load in data
    with open(basis_file,'rb') as myPkl:
        newBasis = pickle.load(myPkl);
    lam = newBasis.eigvals;
    U = newBasis.eigvecs


    myPhi = phi1(FFT_BINS);

    precomp = 0;
    if LOAD_DATA:
        #Load in all PHI data if available
        if os.path.exists(data_file):
            with open(data_file,"rb") as myPkl:
                thisData = pickle.load(myPkl);
            X = thisData.X;
            Y = thisData.Y;
            nsamp, _, _ = np.shape(X);
        else:
            #Since we wanted to load but it doesn't exist, go ahead and generate+save
            all_samples = samples.getAllSamples(Tsub=TSUB,Nsub=NSUB,READ_IN=False) #
            nsamp = len(all_samples);
            extractor = audiolearning.Classifier(myPhi);
            (X,Y) = extractor.extract_features(all_samples)
            thisData = samples.DataPhi();
            thisData.X=X;
            thisData.Y=Y;
            with open(data_file,"wb") as myPkl:
                pickle.dump(thisData,myPkl);
        precomp = 1; #signal we have all data

    #Load in sample framework (not actual data)
    all_samples = samples.getAllSamples(Tsub=TSUB,Nsub=NSUB,READ_IN=False)

    #Divide up the the samples into dictionary for their respective regions
    region_array = dict();
    #Keep track of original order for getting back from X (if loaded)
    orig_index = dict(); k=0;
    for isample in all_samples:
        if isample.region not in region_array.keys():
            region_array[isample.region] = [isample];
            orig_index[isample.region] = [k];
            k+=1;
        else:
            region_array[isample.region].append(isample);
            orig_index[isample.region].append(k);
            k+=1;

    #Count number of samples in each region
    nsamp_region = [len(region_array[ i ]) for i in range(7)];

    #Take certain labels and cluster
    #Set up marker and color for each.
    my_region_markers = {'Rains':['^','b'],'Huang':['o','r'],'Tresidder':['x','g']};
    my_regions = my_region_markers.keys(); #String key
    my_regions_int = [ samples.Regions_dict[i] for i in my_regions ]; #Numbered key

    #How many samples to plot on scatter per region
    nsamp_plot = 15;
    #How many regions are we plotting simultaneously
    nregion_plot = len(my_regions);

    #Initialize empty array with region x sample x reduced_basis
    newCoords = np.zeros((nregion_plot,nsamp_plot,NDIM));

    for (i,iregion) in enumerate(my_regions_int):
        #Shuffle indices to get random samples from each region
        inds = range(nsamp_region[iregion]);
        np.random.shuffle(inds);
        #Select nsamp_plot random samples
        plot_inds = inds[:nsamp_plot];

        for (j,jind) in enumerate(plot_inds):
            if precomp:
                #Use the deterministic order to get data from stored array
                jorig_ind = (orig_index[iregion])[jind]
                jPhi = X[jorig_ind,:,:];
            else:
                #Generate features as you go
                jsample = (region_array[iregion])[jind]
                #Get usual phi
                jPhi = myPhi.get_phi(jsample);

            #Put this sample in new basis
            phi_basis = np.dot(U.T,jPhi.T) #Put subsamples as columns.
            #Average across subsample
            phi_basis = np.mean(phi_basis,1);

            #Add to our matrix
            newCoords[i,j,:] = phi_basis[0:NDIM];

    #Plot
    fig = plt.figure();
    ax = fig.add_subplot(111,projection='3d');

    for (i,iregion) in enumerate(my_regions_int):
        ax.scatter(list(newCoords[i,:,0]),list(newCoords[i,:,1]),list(newCoords[i,:,2]),
                   s = 24, marker = my_region_markers[my_regions[i]][0],
                   c = my_region_markers[my_regions[i]][1],label=my_regions[i]);
    ax.legend();




