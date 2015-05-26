import numpy
from numpy import ma
from scipy.integrate import simps
from scipy import linspace
from scipy import logspace
from scipy import ndimage
import scipy.optimize
import matplotlib.pyplot as plt
import os.path
import pdb
import simtk.unit as units
from pymbar import MBAR
from numpy import linalg
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
import time
import datetime
import sys
from mpl_toolkits.mplot3d import axes3d
import timeseries
import dbscan
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph
from sys import stdout

import gen_rdf_data as genrdf



#################### OPTIONS ####################
relativeErr = False #Create the relative error plot
savedata = True #Save/load dhdl data
load_ukln = True #Save/load u_kln from file
timekln = False #Time energy evaluation, skips actual free energy evaluation
graphsfromfile=True #Generate graphs from the .npz arrays

Ref_state = 1 #Reference state of sampling to pull from

# Set of clustering options
id_regions = True #Run clustering algorithm?
idmethod = 'dbscan' #Choose clustering method, 'lloyd' Lloyd's k-means OR 'dbscan' Density Based Clustering (preferd)
db_rand = True #Boolean. Choose random point inside cluster to sample (True), otherwise use center of mass (False). Only affects 'dbscan' idmethod
################## END OPTIONS ##################




kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
T = 298 * units.kelvin
kT = kB*T
kjpermolTokT = units.kilojoules_per_mole / kT
kjpermolTokcal = 1/4.184

spacing=linspace


#Main output controling vars:
Nparm = 51 #51, 101, or 151
sig_factor=1
savefigs = False
if Nparm == 151 or True:
    alle = True
else:
   alle = False

#Test numpy's savez_compressed routine
try:
    savez = numpy.savez_compressed
except:
    savez = numpy.savez


################ SUBROUTINES ##################

############### Lloyd's Algorithm for voronoi tesalation ############
'''
This block is designed to help me id where the center of a volume of a given feature is
Based on datasicnelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python
'''
import random
def cluster_points(X,mu):
    clusters={}
    for x in X:
        bestmukey = min([(i[0], numpy.linalg.norm(x-mu[i[0]])) for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey]=[x]
    for key in clusters.keys(): #Convert to single array, not a set of arrays
        clusters[key] = numpy.array(clusters[key])
    return clusters
def reevaluate_centers(clusters, weights):
    newmu=[]
    keys= sorted(clusters.keys())
    for k in keys:
        #Convert a set of clusters to their tuple form to feed into the weights
        weight_map = tuple(clusters[k][:,i] for i in range(clusters[k].shape[1])) #shape[1] is the N dim
        #Identify where this cluster is
        tempfeatures = numpy.zeros(weights.shape)
        tempfeatures[weight_map] = 1
        #Find the center of mass based on the weights
        com = numpy.array(ndimage.measurements.center_of_mass(weights, labels=tempfeatures, index=1))
        #Attach these weights to the new mu
        newmu.append(com)
        #newmu.append(numpy.mean(clusters[k],axis=0))
    return newmu
def has_converged(mu,oldmu):
    try:
        converged = (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
    except:
        converged = (set([a for a in mu]) == set([a for a in oldmu]))
    return converged
def find_centers(X, K, weights, verbose=False):
    #Initilize to K random centers
    oldmu =  random.sample(X,K)
    mu = random.sample(X,K)
    #oldmu = [weights[tuple(j for j in i)] for i in oldcord] #Create the list of weights by pulling each "center" and then making the tuple index of the single entry before passing it to weights
    #mu = [weights[tuple(j for j in i)] for i in mucord]
    iteration = 1
    while not has_converged(mu,oldmu):
        if verbose:
            sys.stdout.flush()
            sys.stdout.write('\rIteration: %i' % iteration)
        oldmu=mu
        #assign all points in X to clusters
        clusters = cluster_points(X,mu)
        #Re-evaluate centers
        mu = reevaluate_centers(clusters, weights)
        iteration += 1
    if verbose: sys.stdout.write('\n')
    return mu, clusters
def closest_index(point, features, index):
    #Find the closest point in features at index that is near point.
    #Point should have dimensions to describe a point in features
    #Create array of feature indices
    feature_loc = numpy.where(features == index)
    NFPoints = len(feature_loc[0])
    Ndim = point.shape[0]
    indices = numpy.zeros([Ndim, NFPoints])
    delta = numpy.zeros(indices.shape)
    for dim in range(Ndim):
        indices[dim,:] = feature_loc[dim] #Should grab the array
        delta[dim,:] = feature_loc[dim] - point[dim] #Assign the delta operator
    distance = numpy.sqrt(numpy.sum(delta**2,axis = 0))
    minloc = numpy.where(distance == distance.min())
    return indices[:,minloc]
def closest_point_with_index(point, features):
    #Accepts a point of float "index" and finds the nearest integer index, returns the 
    dims = features.shape
    ndims = len(dims)
    #Round the point
    rpoint = numpy.around(point, out=numpy.zeros(ndims,dtype=int))
    #Since the point should alway be on the interior, i should not have to worry about rounding
    return rpoint, features[tuple(numpy.array(x) for x in rpoint)] #Cast the array to the correct index scheme to return single number not 3 slices of 2-dimensions

def printFreeEnergy(DeltaF_ij, dDeltaF_ij):
    nstates = DeltaF_ij.shape[0]
    print "%s DeltaF_ij:" % "complex"
    for i in range(nstates):
        for j in range(nstates):
            print "%8.3f" % DeltaF_ij[i,j],
        print ""
    print "%s dDeltaF_ij:" % "complex"
    for i in range(nstates):
        for j in range(nstates):
            print "%8.3f" % dDeltaF_ij[i,j],
        print ""
def normalizeData(data):
    data = data.astype(numpy.float)
    min = numpy.min(data)
    max = numpy.max(data)
    return (data - min)/ (max-min)

def find_g_t_states(u_kln, states=None, nequil=None):
    #Subsample multiple states, this assumes you want to subsample independent of what was fed in
    if states is None:
        states = numpy.array(range(nstates))
    num_sample = len(states)
    if nequil is None:
        gen_nequil = True
        nequil = numpy.zeroes(num_sample, dtype=numpy.int32)
    else:
        if len(nequil) != num_sample:
            print "nequil length needs to be the same as length as states!"
            raise
        else:
            gen_nequl = False
    g_t = numpy.zeros([num_sample])
    Neff_max = numpy.zeros([num_sample])
    for state in states:
        g_t[state] = timeseries.statisticalInefficiency(u_kln[k,k,nequil[state]:])
        Neff_max[k] = (u_kln[k,k,:].size + 1) / g_t[state]
    return g_t, Neff_max

def write_g_t(g_t):
    from math import ceil
    ng = len(g_t)
    gfile = open('g_t.txt','w')
    for k in xrange(ng):
        gfile.write('%i\n'% numpy.ceil(g_t[k]))
    gfile.close()

def subsample_series(series, g_t=None, return_g_t=False):
    if g_t is None:
        g_t = timeseries.statisticalInefficiency(series)
    state_indices = timeseries.subsampleCorrelatedData(series, g = g_t, conservative=True)
    N_k = len(state_indices)
    transfer_series = series[state_indices]
    if return_g_t:
        return state_indices, transfer_series, g_t
    else:
        return state_indices, transfer_series

def id_features(data, threshold):
    '''
    Find the locations where features exist inside a data array
    set entries 1=feature, 0=no feature
    Separating these out into individual features is application dependent
    Returns an array of ints={0,1} of data.shape
    '''
    #Filter data. notouch masks covers the sections we are not examining. touch is the sections we want
    data_notouch = ma.masked_less(data, threshold)
    data_touch = ma.masked_greater(data, threshold)
    #Extract the mask to get where there are features. We will use this to id features to operate on
    regions = ma.getmask(data_touch) #Extract the mask from the touch array as the Trues will line up with the areas more than the threshold
    #Create the features map of 1's where we want features (greater than threshold), zeroes otherwise
    features = numpy.zeros(data.shape, dtype=numpy.int32)
    features[regions] = 1 #Define features
    return features

def esq_to_ndx(T, start, end, factor=1, N=51):
    #Quick converter from transformed esq coordinate to effective ndx value
    frac_along = (T**factor - start**factor)/(end**factor-start**factor)
    return frac_along*(N-1)
def ndx_to_esq(ndx, start, end, factor=1, N=51):
    #Quick converter from transformed index to effective esq value
    frac_along = ndx/(N-1)
    esq = (start**factor + (end**factor-start**factor)*frac_along)**(1.0/factor)
    return esq

def check_ceilfloor(vfloor, vceil, N=51):
    if vfloor == vceil:
        if vceil == 0:
            vceil += 1
        else:
            vfloor -= 1
    return vfloor, vceil

def cubemean(point, grid):
    #Triliniear interpolation
    netmean = 0
    px = point[0]
    py = point[1]
    pz = point[2]
    grid_val = []
    x0  = numpy.floor(px)
    x1  = numpy.ceil(px)
    x0,x1 = check_ceilfloor(x0,x1)
    y0  = numpy.floor(py)
    y1  = numpy.ceil(py)
    y0,y1 = check_ceilfloor(y0,y1)
    z0  = numpy.floor(pz)
    z1  = numpy.ceil(pz)
    z0,z1 = check_ceilfloor(z0,z1)
    xd = float(px-x0)/(x1-x0)
    yd = float(py-y0)/(y1-y0)
    zd = float(pz-z0)/(z1-z0)
    c00 = grid[x0,y0,z0]*(1-xd) + grid[x1,y0,z0]*xd
    c10 = grid[x0,y1,z0]*(1-xd) + grid[x1,y1,z0]*xd
    c01 = grid[x0,y0,z1]*(1-xd) + grid[x1,y0,z1]*xd
    c11 = grid[x0,y1,z1]*(1-xd) + grid[x1,y1,z1]*xd
    c0 = c00*(1-yd) + c10*yd
    c1 = c01*(1-yd) + c11*yd
    c = c0*(1-zd) + c1*zd
    return c

def bornG(q, sig, e0, dielec):
    return (q**2 / (4*sig*e0*numpy.pi)) * (1.0/dielec - 1)


class consts(object): #Class to house all constant information

    def _convertunits(self, converter):
        self.const_unaffected_matrix *= converter
        self.const_R_matrix *= converter
        self.const_A_matrix *= converter
        self.const_q_matrix *= converter
        self.const_q2_matrix *= converter
        self.u_kln *= converter
        try:
            self.const_A0_matrix *= converter
            self.const_A1_matrix *= converter
            self.const_R0_matrix *= converter
            self.const_R1_matrix *= converter
            self.const_Un_matrix *= converter
        except:
            pass
        return
    def dimless(self):
        if self.units:
            self._convertunits(kjpermolTokT)
            self.units = False
    def kjunits(self):
        if not self.units:
            self._convertunits(1.0/kjpermolTokT)
            self.units = True

    def save_consts(self, filename):
        savez(filename, u_kln=self.u_kln, const_R_matrix=self.const_R_matrix, const_A_matrix=self.const_A_matrix, const_q_matrix=self.const_q_matrix, const_q2_matrix=self.const_q2_matrix, const_unaffected_matrix=self.const_unaffected_matrix)
    
    def determine_N_k(self, series):
        npoints = len(series)
        #Go backwards to speed up process
        N_k = npoints
        for i in xrange(npoints,0,-1):
            if not numpy.allclose(series[N_k-1:], numpy.zeros(len(series[N_k-1:]))):
                break
            else:
                N_k += -1
        return N_k

    def determine_all_N_k(self, force=False):
        if self.Nkset and not force:
            print "N_k is already set! Use the 'force' flag to manually set it"
            return
        self.N_k = numpy.zeros(self.nstates, dtype=numpy.int32)
        for k in xrange(self.nstates):
            self.N_k[k] = self.determine_N_k(self.u_kln[k,k,:])
        self.Nkset = True
        return

    def updateiter(self, iter):
        if iter > self.itermax:
            self.itermax = iter
    @property #Set the property of itermax which will also update the matricies in place
    def itermax(self):
        return self._itermax
    @itermax.setter #Whenever itermax is updated, the resize should be cast
    def itermax(self, iter):
        if iter > self.itermax:
            ukln_xfer = numpy.zeros([self.nstates, self.nstates, iter])
            unaffected_xfer = numpy.zeros([self.nstates, iter])
            un_xfer = numpy.zeros([self.nstates, iter])
            r0_xfer = numpy.zeros([self.nstates, iter])
            r1_xfer = numpy.zeros([self.nstates, iter])
            r_xfer = numpy.zeros([self.nstates, iter])
            a0_xfer = numpy.zeros([self.nstates, iter])
            a1_xfer = numpy.zeros([self.nstates, iter])
            a_xfer = numpy.zeros([self.nstates, iter])
            q_xfer = numpy.zeros([self.nstates, iter])
            q2_xfer = numpy.zeros([self.nstates, iter])
            #Transfer data
            unaffected_xfer[:,:self.itermax] = self.const_unaffected_matrix
            un_xfer[:,:self.itermax] = self.const_Un_matrix
            a_xfer[:,:self.itermax] = self.const_A_matrix
            r_xfer[:,:self.itermax] = self.const_R_matrix
            q_xfer[:,:self.itermax] = self.const_q_matrix
            q2_xfer[:,:self.itermax] = self.const_q2_matrix
            ukln_xfer[:,:,:self.itermax] = self.u_kln
            self.const_unaffected_matrix = unaffected_xfer
            self.const_R_matrix = r_xfer
            self.const_A_matrix = a_xfer
            self.const_q_matrix = q_xfer
            self.const_q2_matrix = q2_xfer
            self.const_Un_matrix = un_xfer
            self.u_kln = ukln_xfer
            try:
                a0_xfer[:,:self.itermax] = self.const_A0_matrix
                a1_xfer[:,:self.itermax] = self.const_A1_matrix
                r0_xfer[:,:self.itermax] = self.const_R0_matrix
                r1_xfer[:,:self.itermax] = self.const_R1_matrix
                self.const_A0_matrix = a0_xfer
                self.const_A1_matrix = a1_xfer
                self.const_R0_matrix = r0_xfer
                self.const_R1_matrix = r1_xfer
            except:
                pass
            self.shape = self.u_kln.shape
        self._itermax = iter

    def __init__(self, nstates, file=None, itermax=1):
        loaded = False
        self._itermax=itermax
        self.nstates=nstates
        self.Nkset = False
        if file is not None:
            try:
                ukln_file = numpy.load(file)
                self.u_kln = ukln_file['u_kln']
                self.const_R_matrix = ukln_file['const_R_matrix'] 
                self.const_A_matrix = ukln_file['const_A_matrix'] 
                self.const_unaffected_matrix = ukln_file['const_unaffected_matrix']
                self.const_q_matrix = ukln_file['const_q_matrix']
                self.const_q2_matrix = ukln_file['const_q2_matrix']
                self._itermax = self.u_kln.shape[2]
                self.determine_all_N_k()
                self.Nkset = True
                loaded = True
            except: 
                pass
        if not loaded:
            self.const_unaffected_matrix = numpy.zeros([self.nstates,self.itermax])
            self.const_Un_matrix = numpy.zeros([self.nstates,self.itermax])
            self.const_R0_matrix = numpy.zeros([self.nstates,self.itermax])
            self.const_R1_matrix = numpy.zeros([self.nstates,self.itermax])
            self.const_R_matrix = numpy.zeros([self.nstates,self.itermax])
            self.const_A0_matrix = numpy.zeros([self.nstates,self.itermax])
            self.const_A1_matrix = numpy.zeros([self.nstates,self.itermax])
            self.const_A_matrix = numpy.zeros([self.nstates,self.itermax])
            self.const_q_matrix = numpy.zeros([self.nstates,self.itermax])
            self.const_q2_matrix = numpy.zeros([self.nstates,self.itermax])
            self.u_kln = numpy.zeros([self.nstates,self.nstates,self.itermax])
            self.itermax = itermax
            self.N_k = numpy.ones(self.nstates) * self.itermax
        self.shape = self.u_kln.shape
        self.units = True

def execute(nstates, q_samp_space, epsi_samp_space, sig_samp_space, singlestate=None, singleparms=None, useO=False, crunchset=False):
    #Initilize limts
    sig3_samp_space = sig_samp_space**3
    #These are the limits used to compute the constant matricies
    #They should match with LJ 0 and 5, ALWAYS
    q_min = -2.0
    q_max = +2.0
    epsi_min = epsi_samp_space[0]
    epsi_max = epsi_samp_space[5]
    sig_min = sig_samp_space[0]
    sig_max = sig_samp_space[5]
    lamto_epsi = lambda lam: (epsi_max - epsi_min)*lam + epsi_min
    lamto_sig3 = lambda lam: (sig_max**3 - sig_min**3)*lam + sig
    lamto_sig = lambda lam: lamto_sig3(lam)**(1.0/3)
    if spacing is logspace:
        StartSpace = -5
        EndSpace   = 0
        spacename='log'
        PlotStart = 10**StartSpace
        PlotEnd   = 10**EndSpace
    elif spacing is linspace:
        sigStartSpace = sig_min
        sigEndSpace   = sig_max
        epsiStartSpace = epsi_min
        qStartSpace = q_min
        qEndSpace = q_max
        if alle:
            epsiEndSpace   = 3.6 #!!! Manual set
        else:
            epsiEndSpace   = epsi_max
        spacename='linear'
        sigPlotStart = sigStartSpace**sig_factor
        sigPlotEnd   = sigEndSpace**sig_factor
        epsiPlotStart = epsiStartSpace
        epsiPlotEnd   = epsiEndSpace
        qPlotStart = qStartSpace
        qPlotEnd = qEndSpace



    #generate sample length
    g_en_start = 19 #Row where data starts in g_energy output
    g_en_energy = 1 #Column where the energy is located
    niterations_max = 30001
    #Min and max sigmas:
    fC12 = lambda epsi,sig: 4*epsi*sig**12
    fC6 = lambda epsi,sig: 4*epsi*sig**6
    fsig = lambda C12, C6: (C12/C6)**(1.0/6)
    fepsi = lambda C12, C6: C6**2/(4*C12)
    C12_delta = fC12(epsi_max, sig_max) - fC12(epsi_min, sig_min)
    C6_delta = fC6(epsi_max, sig_max) - fC6(epsi_min, sig_min)
    C12_delta_sqrt = fC12(epsi_max, sig_max)**.5 - fC12(epsi_min, sig_min)**.5
    C6_delta_sqrt = fC6(epsi_max, sig_max)**.5 - fC6(epsi_min, sig_min)**.5
    #Set up lambda calculation equations
    flamC6 = lambda epsi, sig: (fC6(epsi, sig) - fC6(epsi_min, sig_min))/C6_delta
    flamC12 = lambda epsi, sig: (fC12(epsi, sig) - fC12(epsi_min, sig_min))/C12_delta
    flamC6sqrt = lambda epsi, sig: (fC6(epsi, sig)**.5 - fC6(epsi_min, sig_min)**.5)/C6_delta_sqrt
    flamC12sqrt = lambda epsi, sig: (fC12(epsi, sig)**.5 - fC12(epsi_min, sig_min)**.5)/C12_delta_sqrt
    flamC1 = lambda q: q
    lamC12 = flamC12sqrt(epsi_samp_space, sig_samp_space)
    lamC6 = flamC6sqrt(epsi_samp_space, sig_samp_space)
    lamC1 = flamC1(q_samp_space)
    #Try to load u_kln
    lam_range = linspace(0,1,nstates)
    subsampled = numpy.zeros([nstates],dtype=numpy.bool)
    if load_ukln and os.path.isfile('esq_ukln_consts_n%i.npz'%nstates):
        energies = consts(nstates, file='esq_ukln_consts_n%i.npz'%nstates)
    else:
        #Initial u_kln
        energies = consts(nstates)
        g_t = numpy.zeros([nstates])
        #Read in the data
        for k in xrange(nstates):
            print "Importing LJ = %02i" % k
            energy_dic = {'full':{}, 'rep':{}}
            #Try to load the subsampled filenames
            try:
                energy_dic['null'] = open('lj%s/prod/subenergy%s_null.xvg' %(k,k),'r').readlines()[g_en_start:] #Read in the null energies (unaffected) of the K states
                for l in xrange(nstates):
                    energy_dic['full']['%s'%l] = open('lj%s/prod/subenergy%s_%s.xvg' %(k,k,l),'r').readlines()[g_en_start:] #Read in the full energies for each state at KxL
                    if l == 5 or l == 0:
                        energy_dic['rep']['%s'%l] = open('lj%s/prod/subenergy%s_%s_rep.xvg' %(k,k,l),'r').readlines()[g_en_start:] #Read in the repulsive energies at 0, nstates-1, and K
                energy_dic['q']  =  open('lj%s/prod/subenergy%s_q.xvg' %(k,k),'r').readlines()[g_en_start:] #Read in the charge potential energy
                energy_dic['q2']  =  open('lj%s/prod/subenergy%s_q2.xvg' %(k,k),'r').readlines()[g_en_start:] #Read in the charge potential energy
                iter = len(energy_dic['null'])
                subsampled[k] = True
                # Set the object to iterate over, since we want every frame of the subsampled proces, we just use every frame
                frames = xrange(iter)
            except: #Load the normal way
                energy_dic['null'] = open('lj%s/prod/energy%s_null.xvg' %(k,k),'r').readlines()[g_en_start:] #Read in the null energies (unaffected) of the K states
                for l in xrange(nstates):
                    energy_dic['full']['%s'%l] = open('lj%s/prod/energy%s_%s.xvg' %(k,k,l),'r').readlines()[g_en_start:] #Read in the full energies for each state at KxL
                    if l == 5 or l == 0:
                        energy_dic['rep']['%s'%l] = open('lj%s/prod/energy%s_%s_rep.xvg' %(k,k,l),'r').readlines()[g_en_start:] #Read in the repulsive energies at 0, nstates-1, and K
                energy_dic['q']  =  open('lj%s/prod/energy%s_q.xvg' %(k,k),'r').readlines()[g_en_start:] #Read in the charge potential energy
                energy_dic['q2']  =  open('lj%s/prod/energy%s_q2.xvg' %(k,k),'r').readlines()[g_en_start:] #Read in the charge potential energy
                iter = niterations_max
                #Subsample
                tempenergy = numpy.zeros(iter)
                for frame in xrange(iter):
                    tempenergy[frame] = float(energy_dic['full']['%s'%k][frame].split()[g_en_energy])
                frames, temp_series, g_t[k] = subsample_series(tempenergy, return_g_t=True)
                print "State %i has g_t of %i" % (k, g_t[k])
                iter = len(frames)
            #Update iterations if need be
            energies.updateiter(iter)
            #Fill in matricies
            n = 0
            for frame in frames:
                #Unaffected state
                energies.const_Un_matrix[k,n] = float(energy_dic['null'][frame].split()[g_en_energy])
                #Charge only state
                VI = float(energy_dic['q'][frame].split()[g_en_energy]) - energies.const_Un_matrix[k,n]
                VII = float(energy_dic['q2'][frame].split()[g_en_energy]) - energies.const_Un_matrix[k,n]
                q1 = 1
                q2 = 0.5
                QB = (q1**2*VII - q2**2*VI)/(q1**2*q2 - q2**2*q1)
                QA = (VI - q1*QB)/q1**2
                energies.const_q_matrix[k,n] = QB
                energies.const_q2_matrix[k,n] = QA
                #const_q_matrix[k,n] = float(energy_dic['q'][n].split()[g_en_energy]) - const_Un_matrix[k,n]
                #Isolate the data
                for l in xrange(nstates):
                    energies.u_kln[k,l,n] = float(energy_dic['full']['%s'%l][frame].split()[g_en_energy]) #extract the kln energy, get the line, split the line, get the energy, convert to float, store
                #Repulsive terms: 
                #R0 = U_rep[k,k,n] + dhdl[k,0,n] - Un[k,n]
                energies.const_R0_matrix[k,n] = float(energy_dic['rep']['%s'%(0)][frame].split()[g_en_energy]) - energies.const_Un_matrix[k,n]
                #R1 = U_rep[k,k,n] + dhdl[k,-1,n] - Un[k,n]
                energies.const_R1_matrix[k,n] = float(energy_dic['rep']['%s'%(5)][frame].split()[g_en_energy]) - energies.const_Un_matrix[k,n]
                energies.const_R_matrix[k,n] = energies.const_R1_matrix[k,n] - energies.const_R0_matrix[k,n]
                #Finish the total unaffected term
                #Total unaffected = const_Un + U0 = const_Un + (U_full[k,0,n] - const_Un) = U_full[k,0,n]
                energies.const_unaffected_matrix[k,n] = energies.u_kln[k,0,n]
                #Fill in the q matrix
                
                #Attractive term
                #u_A = U_full[k,n] - constR[k,n] - const_unaffected[k,n]
                energies.const_A0_matrix[k,n] = energies.u_kln[k,0,n] - energies.const_R0_matrix[k,n] - energies.const_Un_matrix[k,n]
                energies.const_A1_matrix[k,n] = energies.u_kln[k,5,n] - energies.const_R1_matrix[k,n] - energies.const_Un_matrix[k,n]
                energies.const_A_matrix[k,n] = energies.const_A1_matrix[k,n] - energies.const_A0_matrix[k,n]
                n += 1
        energies.determine_all_N_k()
        #write_g_t(g_t)
        constname = 'esq_ukln_consts_n%i.npz'%nstates
        if load_ukln and not os.path.isfile(constname):
            energies.save_consts(constname)
    #Sanity check
    sanity_kln = numpy.zeros(energies.u_kln.shape)
    #Round q to the the GROMACS precision, otherwise there is drift in q which can cause false positives
    lamC1r = numpy.around(lamC1, decimals=4)
    for l in xrange(nstates):
        #sanity_kln[:,l,:] = lamC12[l]*energies.const_R_matrix + lamC6[l]*energies.const_A_matrix + lamC1[l]*energies.const_q_matrix + lamC1[l]**2*energies.const_q2_matrix + energies.const_unaffected_matrix
        sanity_kln[:,l,:] = lamC12[l]*energies.const_R_matrix + lamC6[l]*energies.const_A_matrix + lamC1r[l]*energies.const_q_matrix + lamC1r[l]**2*energies.const_q2_matrix + energies.const_unaffected_matrix
    del_kln = numpy.abs(energies.u_kln - sanity_kln)
    del_tol = 1 #in kJ per mol
    print "Max Delta: %f" % numpy.nanmax(del_kln)
    if numpy.nanmax(del_kln) > 1: #Check for numeric error
        #Double check to see if the weight is non-zero.
        #Most common occurance is when small particle is tested in large particle properties
        #Results in energies > 60,000 kj/mol, which carry 0 weight to machine precision
        nonzero_weights = numpy.count_nonzero(numpy.exp(-energies.u_kln[numpy.where(del_kln > .2)] * kjpermolTokT))
        if nonzero_weights != 0:
            print "and there are %d nonzero weights! Stopping execution" % nonzero_weights
            pdb.set_trace()
        else:
            print "but these carry no weight and so numeric error does not change the answer"

    ##################################################
    ############### END DATA INPUT ###################
    ##################################################
    #Convert to dimless
    energies.dimless()
    #Set up regular grid
    sig_range = (spacing(sigStartSpace**3,sigEndSpace**3,Nparm))**(1.0/3)
    epsi_range = spacing(epsiStartSpace,epsiEndSpace,Nparm)
    q_range = spacing(qStartSpace,qEndSpace,Nparm)
    epsi_plot_range = spacing(epsiStartSpace,epsi_max,Nparm)
    #Load subsequent f_ki
    f_ki_loaded = False
    state_counter = nstates
    while not f_ki_loaded and state_counter != 0:
        #Load the largest f_ki you can
        try:
            f_ki_load = numpy.load('esq_f_k_{myint:{width}}.npy'.format(myint=state_counter, width=len(str(state_counter))))
            f_ki_loaded = True
            f_ki_n = state_counter
        except:
            pass
        state_counter -= 1
    try:
        if nstates >= f_ki_n:
            draw_ki = f_ki_n
        else:
            draw_ki = nstates
        #Copy the loaded data
        f_ki = numpy.zeros(nstates)
        f_ki[:draw_ki] = f_ki_load[:draw_ki]
        mbar = MBAR(energies.u_kln, energies.N_k, verbose = True, initial_f_k=f_ki, subsampling_protocol=[{'method':'L-BFGS-B','options':{'disp':True}}], subsampling=1)
    except:
        mbar = MBAR(energies.u_kln, energies.N_k, verbose = True, subsampling_protocol=[{'method':'L-BFGS-B','options':{'disp':True}}], subsampling=1)
    if not f_ki_loaded or f_ki_n != nstates:
        try:
            numpy.save_compressed('esq_f_k_{myint:{width}}.npy'.format(myint=nstates, width=len(str(nstates))), mbar.f_k)
        except:
            numpy.save('esq_f_k_{myint:{width}}.npy'.format(myint=nstates, width=len(str(nstates))), mbar.f_k)

    ######## Begin Computing RDF's #########

    basebins = genrdf.defbin
    plotx = linspace(genrdf.distmin, genrdf.distmax, basebins)/10.0 #put in nm
    #Generate blank
    rdfs = numpy.zeros([basebins, nstates, energies.itermax], dtype=numpy.float64)
    if useO:
        pathstr = 'lj%s/prod/rdfOhist%s.npz'
    else:
        pathstr = 'lj%s/prod/rdfhist%s.npz'
    for k in xrange(nstates):
        try: #Try to load in
            if useO:
                rdfk = numpy.load(pathstr%(k,k))['rdf']
            else:
                rdfk = numpy.load(pathstr%(k,k))['rdf']
        except:
            rdfk = genrdf.buildrdf(k, useO=useO)
        nkbin, nkiter = rdfk.shape
        rdfs[:,k,:nkiter] = rdfk
  
    if singleparms is not None:
        Npert = 1
        pertq = numpy.array([singleparms[0]])
        pertepsi = numpy.array([singleparms[1]])
        pertsig = numpy.array([singleparms[2]])
        pertrdfs = None
        if useO:
            filestring = 'pertrdfs/pertrdfOn%s' % nstates
        else:
            filestring = 'pertrdfs/pertrdfn%s' % nstates
        filestring += 'q%fe%fs%f.npz'
        fileobjs = [singleparms[0],singleparms[1],singleparms[2]] #Even though 0 and 1 get +l, the l =0 in this instance so its fine
    elif singlestate is not None:
        Npert = 1
        pertq = numpy.array([q_samp_space[singlestate]])
        pertepsi = numpy.array([epsi_samp_space[singlestate]])
        pertsig = numpy.array( [sig_samp_space[singlestate]])
        try:
            pertrdf = numpy.load(pathstr%(singlestate,singlestate))['rdf']
        except:
            pertrdf = genrdf.buildrdf(singlestate, useO=useO)
        pertbin, pertnk = pertrdf.shape
        quickrdf = numpy.zeros([pertbin,1,pertnk]) #Initilze the matrix I will pass into the graphing routine
        quickrdf[:,0,:] = pertrdf
        pertrdf = quickrdf
        del quickrdf #Cleanup a bit
        if useO:
            filestring = 'lj%s/rdfO%sfromn%s.npz'
        else:
            filestring = 'lj%s/rdf%sfromn%s.npz'
        fileobjs = [singlestate,singlestate,nstates]
    elif crunchset is True:
        #Joung/Cheatham params: 
        pertname =             ['Li+',      'Na+',       'K+',        'Rb+',       'Cs+',       'F-',        'Cl-',       'Br-',       'I-']
        pertq =    numpy.array([1,          1,           1,           1,           1,           -1,          -1,          -1,          -1])
        pertepsi = numpy.array([0.11709719,  0.365846031, 0.810369254, 1.37160683,  1.70096085,  0.014074976, 0.148912744, 0.245414194, 0.224603814])
        pertsig =  numpy.array([0.196549675, 0.24794308,  0.30389715,  0.323090391, 0.353170773, 0.417552588, 0.461739606, 0.482458908, 0.539622469])
        Npert = len(pertq)
        if useO:
            filestring = 'pertrdfs/pertrdfOn%s%s.npz'
        else:
            filestring = 'pertrdfs/pertrdfn%s%s.npz'
    else:
        #For now, just use the back 5 of the qes set
        Npert = 5
        pertq = q_samp_space[-Npert:]
        pertepsi = epsi_samp_space[-Npert:]
        pertsig = sig_samp_space[-Npert:]
        pertNk = energies.N_k[-Npert:]
        pertrdfs = rdfs[:,-Npert:,:]
        if useO:
            filestring = 'lj%s/rdfO%sfromn%s.npz'
        else:
            filestring = 'lj%s/rdf%sfromn%s.npz'
        fileobjs = [nstates-Npert,nstates-Npert,nstates]

    
    #pertq = numpy.array([q_samp_space[1]])+0.25
    #pertepsi = numpy.array([epsi_samp_space[1]])
    #pertepsi *= 2
    #pertsig = numpy.array([sig_samp_space[1]])
    #pertsig *= 1.5
    #pertNk = numpy.array([energies.N_k[1]])
    #pertrdfs = rdfs[:,1,:]
 

    #Begin computing expectations
    Erdfs = numpy.zeros([Npert, basebins])
    dErdfs = numpy.zeros([Npert, basebins])
    
    for l in xrange(Npert):
        #if not savedata or not os.path.isfile('lj%s/rdf%sfromn%s.npz'%(nstates-Npert+l, nstates-Npert+l, nstates)):
        if crunchset:
            filecheck = filestring % (nstates, pertname[l])
        else:
            filecheck = filestring % (fileobjs[0]+l, fileobjs[1]+l, fileobjs[2])
        if not savedata or not os.path.isfile(filecheck):
            print "Working on state %d/%d" %(l+1, Npert)
            q = pertq[l]
            epsi = pertepsi[l]
            sig = pertsig[l]
            #u_kn_P = numpy.zeros([nstates,energies.itermax])
            u_kn_P = flamC12sqrt(epsi,sig)*energies.const_R_matrix + flamC6sqrt(epsi,sig)*energies.const_A_matrix + flamC1(q)*energies.const_q_matrix + flamC1(q)**2*energies.const_q2_matrix + energies.const_unaffected_matrix
            for bin in xrange(basebins):
                stdout.flush()
                stdout.write('\rWorking on bin %d/%d'%(bin,basebins-1))
                Erdfs[l, bin], dErdfs[l, bin] = mbar.computePerturbedExpectation(u_kn_P, rdfs[bin,:,:])
            stdout.write('\n')
            if savedata:
                #savez('lj%s/rdf%sfromn%s.npz'%(nstates-Npert+l, nstates-Npert+l, nstates), Erdfs=Erdfs[l,:], dErdfs=dErdfs[l,:])
                if crunchset:
                    savez(filestring % (nstates, pertname[l]), Erdfs=Erdfs[l,:], dErdfs=dErdfs[l,:])
                else:
                    savez(filestring % (fileobjs[0]+l, fileobjs[1]+l, fileobjs[2]), Erdfs=Erdfs[l,:], dErdfs=dErdfs[l,:])
        else:
            #rdfdata = numpy.load('lj%s/rdf%sfromn%s.npz'%(nstates-Npert+l, nstates-Npert+l, nstates))
            if crunchset:
                rdfdata = numpy.load(filestring % (nstates, pertname[l]))
            else:
                rdfdata = numpy.load(filestring % (fileobjs[0]+l, fileobjs[1]+l, fileobjs[2]))
            Erdfs[l,:] = rdfdata['Erdfs']
            dErdfs[l,:] = rdfdata['dErdfs']

    if singleparms is None and not crunchset:
        f,a = plt.subplots(Npert, 2)
        for l in xrange(Npert):
            Nl = pertNk[l]
            rdftraj = pertrdfs[:,l,:Nl]
            if Npert == 1:
                a[0].plot(plotx, Erdfs[l,:], '-k')
                #a[0].plot(plotx, Erdfs[l,:] + dErdfs[l,:], '--k')
                #a[0].plot(plotx, Erdfs[l,:] - dErdfs[l,:], '--k')
                a[1].plot(plotx, rdftraj.sum(axis=1)/float(Nl), '-k')
                alims0 = a[0].get_ylim()
                alims1 = a[1].get_ylim()
                a[0].set_ylim(0, numpy.amax(numpy.array([alims0[1], alims1[1]])))
                a[1].set_ylim(0, numpy.amax(numpy.array([alims0[1], alims1[1]])))
            else:
                a[l,0].plot(plotx, Erdfs[l,:], '-k')
                #a[l,0].plot(plotx, Erdfs[l,:] + dErdfs[l,:], '--k')
                #a[l,0].plot(plotx, Erdfs[l,:] - dErdfs[l,:], '--k')
                a[l,1].plot(plotx, rdftraj.sum(axis=1)/float(Nl), '-k')
                alims0 = a[l,0].get_ylim()
                alims1 = a[l,1].get_ylim()
                a[l,0].set_ylim(0, numpy.amax(numpy.array([alims0[1], alims1[1]])))
                a[l,1].set_ylim(0, numpy.amax(numpy.array([alims0[1], alims1[1]])))
    elif crunchset:
        f,a = plt.subplots(Npert)
        for l in xrange(Npert):
            a[l].plot(plotx, Erdfs[l,:], '-k')
            a[l].set_ylim([0,a[l].get_ylim()[1]])
            print "Peak for %s at %f" % (pertname[l], plotx[numpy.argmax(Erdfs[l,:])])
    else:
        f,a=plt.subplots(1,1)
        a.plot(plotx, Erdfs[0,:], '-k')
        
    plt.show()
    pdb.set_trace()
     

    ################################################
    ################# PLOTTING #####################
    ################################################

####################################################################################
####################################################################################
####################################################################################

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--nstates", dest="nstates", default=None, help="Set the number of states", metavar="NSTATES")
    parser.add_option("-t", "--truncate", action='store_true', dest="truncate", default=False, help="Truncate the input qes", metavar="TRUNC")
    parser.add_option("--single", dest="single", default=None, help="Single index to estimate at", metavar="SINGLE")
    parser.add_option("--set", dest="set", action='store_true', default=False, help="Run through the set of parameters", metavar="SET")
    parser.add_option("--sq", "--singleq", dest="singleq", default=None, type='float', help="Single q to estimate at, also needs e and s", metavar="SINGLEQ")
    parser.add_option("--se", "--singlee", dest="singlee", default=None, type='float', help="Single e to estimate at, also needs q and s", metavar="SINGLEE")
    parser.add_option("--ss", "--singles", dest="singles", default=None, type='float', help="Single s to estimate at, also needs e and q", metavar="SINGLES")
    parser.add_option("-O", "--oxygen", action='store_true', dest="oxygen", default=False, help="Estimate RDF from oxygen only", metavar="OXYGEN")
    (options, args) = parser.parse_args()
    x = numpy.load('qes.npy')
    qs = x[:,0]
    es = x[:,1]
    ss = x[:,2]
    if options.nstates is None:
        nstates = len(qs)
    else:
        nstates = int(options.nstates)
        if options.truncate and (options.single is not None):
            qs = qs[:nstates]
            es = es[:nstates]
            ss = ss[:nstates]
    if (options.singleq is not None) and (options.singlee is not None) and (options.singles is not None):
        singleparms = [options.singleq, options.singlee, options.singles]
    else:
        singleparms = None
    #If this script is run with manaual execution, break here since its probably being run as debugging anyways.
    pdb.set_trace()
    execute(nstates, qs, es, ss, singlestate=options.single, singleparms=singleparms, useO=options.oxygen, crunchset=options.set)
