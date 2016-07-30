import numpy
#from scipy import linspace
import timeseries
import pdb
#import matplotlib.pyplot as plt
import sys
import math

def subsample_series(series, g_t=None, return_g_t=False):
    if g_t is None:
        g_t = timeseries.statisticalInefficiency(series)
    state_indices = timeseries.subsampleCorrelatedData(series, g = g_t, conservative=True)
    N_k = len(state_indices)
    transfer_series = series[state_indices]
    if return_g_t:
        return state_indices, transfer_series, int(math.ceil(g_t))
    else:
        return state_indices, transfer_series

if __name__ == "__main__":
    g_en_start = 19 #Row where data starts in g_energy output
    g_en_energy = 1 #Column where the energy is located
    narg = len(sys.argv)
    try:
        startk = int(sys.argv[1])
    except:
        startk = 36
    try:
        endk = int(sys.argv[2])
    except:
        endk = 46
    
    for k in xrange(startk,endk):
        energy = open('/scratch/ln8dc/ljsphere_es/lj{0:d}/prod/base_energy{0:d}.xvg'.format(k),'r').readlines()[g_en_start:] #Read in the base energy
        iter = len(energy)
        tempenergy = numpy.zeros(iter)
        for frame in xrange(iter):
            tempenergy[frame] = float(energy[frame].split()[g_en_energy])
        #f,a = plt.subplots(1,1)
        #a.plot(xrange(iter), tempenergy)
        #plt.show()
        frames, temp_series, g_t = subsample_series(tempenergy, return_g_t=True)
        #print "State %i has g_t of %i" % (k, g_t)
        gfile = open('/scratch/ln8dc/ljsphere_es/lj{0:d}/prod/g_t{0:d}.txt'.format(k),'w')
        gfile.write('{0:d}'.format(g_t))
        gfile.close()
