# -*- coding: utf-8 -*-
"""
Created on Sat May 21 22:22:28 2016

@author: ReidW
"""
import time

import samples
reload(samples)
import spectral
reload(spectral)
import audiolearning
reload(audiolearning)

def tic():
    #Homemade version of matlab tic and toc functions

    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

tic()
all_samples = samples.getAllSamples(T=3,N=1,key="phone",val="Reid")
spectral.getAllSPED(all_samples,20);
toc()

