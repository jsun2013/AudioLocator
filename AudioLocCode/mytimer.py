# -*- coding: utf-8 -*-
"""
Created on Sun May 22 14:52:29 2016

@author: ReidW
"""


import time
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