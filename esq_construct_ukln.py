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

'''
This code is written to support specific GROMACS energy output files for single ion parameters in epsilon, sigma, and q space.
Simulations run must be fully geometric mixing rules for both sigma_ij and epsilon_ij

Main call is "execute(Nstates, q_samp_space, epsi_samp_space, sig_samp_space)" which expects:
    Nstates : integer
        number of sampled states that are used in this calculation
    q_samp_space : 1D iterable of len >= Nstates
        Sampled partial charges of the particle in units of elementary charge
    epsi_samp_space : 1D iterable of len >= Nstates
        Sampled particle epsilon_ii in units of kJ/mol
    sig_samp_space : 1D iterable of len >= Nstates
        Sampled particle sigma_ii in units of nm
    
    Note: q_- epsi_- and sig_samp_space can all be longer than Nstates, the code will just stop looking past the Nstates'th index
'''



#################### OPTIONS ####################
relativeErr = False #Create the relative error plot, not very helpful since unknown errors can lead to NaNs in the graphs
savedata = True #Save/load dhdl data
load_ukln = True #Save/load u_kln from file
timekln = False #Time energy evaluation, skips actual free energy evaluation
graphsfromfile=True #Generate graphs from the .npz arrays

Ref_state = 1 #Index of reference state of sampling to pull from.

# Set of clustering options
id_regions = True #Run clustering algorithm?
idmethod = 'dbscan' #Choose clustering method, 'lloyd' Lloyd's k-means OR 'dbscan' Density Based Clustering (preferd)
db_rand = True #Boolean. Choose random point inside cluster to sample (True), otherwise use center of mass (False). Only affects 'dbscan' idmethod
################## END OPTIONS ##################



NA = units.AVOGADRO_CONSTANT_NA
kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
T = 298 * units.kelvin
kT = kB*T
kjpermolTokT = units.kilojoules_per_mole / kT
kjpermolTokcal = 1/4.184

#Used to also support logarithmic spacing, now is just linear spacing
spacing=linspace


#output controling vars mostly legacy but some still in use:
Nparm = 51 #51, 101, or 151
sig_factor=1 #What scaling factor to use on sigma, either sigma^1 or simga^3
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

def G_ion_cav(R, energy='kcalpermol'):
    #Function to return dimensionless free energy in terms of cavitation free energy, just feed it an R with units
    #Some water properties curtosey of Hunenberger and Reif
    water_gamma = 71.99E-3 * units.joules / units.meter**2 #Surface Tension
    P = 1*units.atmosphere #Pressure my simulations were run at
    G_units =  (4*numpy.pi*NA * (water_gamma*R**2 + (1.0/3)*P*R**3))
    if energy is 'kcalpermol':
        G_out = G_units / units.kilocalories_per_mole
    elif energy is 'kjpermol' or energy is 'kJpermol':
        G_out = G_units / units.kilojoules_per_mole
    elif energy is 'kT' or energy is 'kt':
        G_out = G_units / units.kilojoules_per_mole * kjpermolTokT
    else:
        G_out = G_units 
    return G_out

class consts(object): 
    '''
    Keep all basis function data and operations housed in a single location. Prevents user from having to manipulate every basis function energy.
    '''
#Class to house all constant information

    def _convertunits(self, converter):
        #Flip between unit set. Probably could be replaced with cleaner function
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
        #Create a npz array to save data for rapid loading on new run
        savez(filename, u_kln=self.u_kln, const_R_matrix=self.const_R_matrix, const_A_matrix=self.const_A_matrix, const_q_matrix=self.const_q_matrix, const_q2_matrix=self.const_q2_matrix, const_unaffected_matrix=self.const_unaffected_matrix)
    
    def determine_N_k(self, series):
        #Determine how many samples there are in a given series, see the determine_all_N_k for useage
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
        #Figure out how many samples there are in each k'th state by going backwards from the last index to find the first non-zero entry.
        #This will miss any entries that actually are 0 energy on the tail though, but these are rare enough that you may want to recode this function anyways.
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
        '''
        Primary call for this class. If a file is given, it will attempt to load all basis function data from the file, expecting the same names that self.save_consts would write out.
        '''
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

def execute(nstates, q_samp_space, epsi_samp_space, sig_samp_space):
    #Initilize limts
    sig3_samp_space = sig_samp_space**3
    #These are the limits used to compute the constant matricies
    #They should match with LJ 0 and 5, ALWAYS
    q_min = -2.0
    q_max = +2.0
    #Set the 2 main A and B states that the basis function energies are computed with respect to
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
            epsiEndSpace   = 3.6 #Manual set, I chose this value as it looks good for the range I was searching
        else:
            epsiEndSpace   = epsi_max
        spacename='linear'
        sigPlotStart = sigStartSpace**sig_factor
        sigPlotEnd   = sigEndSpace**sig_factor
        epsiPlotStart = epsiStartSpace
        epsiPlotEnd   = epsiEndSpace
        qPlotStart = qStartSpace
        qPlotEnd = qEndSpace


    '''
    Start data input

    Expects GROMACS style data outputs
    '''
    #generate sample length
    g_en_start = 19 #Row where data starts in g_energy output
    g_en_energy = 1 #Column where the energy is located
    niterations_max = 30001
    #Min and max sigmas, mostly helper functions:
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
    ################################################################
    ######## Optional block: specific free energy estimates ########
    ################################################################
    ##Joung and Chetham Parameters
    #ion_names =             ['Li+',      'Na+',       'K+',        'Rb+',       'Cs+',       'F-',        'Cl-',       'Br-',       'I-']
    #ion_qs =    numpy.array([1,           1,           1,           1,           1,           -1,          -1,          -1,          -1])
    #ion_epsis = numpy.array([0.11709719,  0.365846031, 0.810369254, 1.37160683,  1.70096085,  0.014074976, 0.148912744, 0.245414194, 0.224603814])
    #ion_sig =   numpy.array([0.196549675, 0.24794308,  0.30389715,  0.323090391, 0.353170773, 0.417552588, 0.461739606, 0.482458908, 0.539622469])
    #ion_N = len(ion_names)

    #spec_DelF = numpy.zeros(ion_N)
    #spec_dDelF = numpy.zeros(ion_N)
    #u_kln_P = numpy.zeros([nstates,ion_N + 1,energies.itermax])
    #u_kln_P[:,0,:] = energies.u_kln[:,Ref_state,:]
    #for i in xrange(ion_N):
    #    epsi = ion_epsis[i]
    #    sig = ion_sig[i]
    #    q = ion_qs[i]
    #    u_kln_P[:,i+1,:] = flamC12sqrt(epsi,sig)*energies.const_R_matrix + flamC6sqrt(epsi,sig)*energies.const_A_matrix + flamC1(q)*energies.const_q_matrix + flamC1(q)**2*energies.const_q2_matrix + energies.const_unaffected_matrix
    #(DeltaF_ij, dDeltaF_ij) = mbar.computePerturbedFreeEnergies(u_kln_P, uncertainty_method='svd-ew-kab')
    #for i in xrange(ion_N):
    #    #Unwrap the data, reference state is in state 0
    #    spec_DelF[i] = DeltaF_ij[0, 1+i]
    #    spec_dDelF[i] = dDeltaF_ij[0, 1+i]
    #spec_DelF *= kjpermolTokcal/kjpermolTokT
    #spec_dDelF *= kjpermolTokcal/kjpermolTokT
    #for i in xrange(ion_N):
    #    print "Ion %s has relative FE of: %f +- %f kcal/mol" %(ion_names[i], spec_DelF[i],spec_dDelF[i])
    #pdb.set_trace()
    ################################################################
    ###################### END OPTIONAL BLOCK ######################
    ################################################################
    

    ######## #Begin computing free energies ##########
    '''
    Check a few things before attemping free energy:
    1) It looks for the Nparm**3 DelF file or the Nparm count of Nparm**2 files (more detailed). Instead of checking for all Nparm detailed files, it looks for the last one since all the others would have to be written first
    2) If the timekln flag is set, it will run the energy evaluation routine, but not the free energy block to time how long it takes to compute the energies over the Nparm**3 space
    3) Since spacing can technically be done either in sigma or in sigma**3, it checks for either one.
    '''
    if not (os.path.isfile('es_freeEnergies%s.npz'%spacename) and graphsfromfile) or not (os.path.isfile('esq_%s/ns%iNp%iQ%i.npz' % (spacename, nstates, Nparm, Nparm-1)) and savedata) or timekln: #nand gate +timing flag
        #Create numpy arrys: q, epsi, sig
        DelF = numpy.zeros([Nparm, Nparm, Nparm])
        dDelF = numpy.zeros([Nparm, Nparm, Nparm])
        #Populate energies
        run_start_time = time.time()
        number_of_iterations = Nparm
        iteration = 0
        number_of_iterations = Nparm
        for iq in xrange(Nparm):
            #Grab charge
            q = q_range[iq]
            initial_time = time.time()
            iteration += 1
            print "Q index: %i/%i" % (iq, Nparm-1)
            #Using PerturpedFreeEnergies instead of recreating the MBAR object every time. Saves time with same accuracy
            #Perturbed assumes all l states are unsampled
            #Perturbed free energies can only be computed up to a separate additive constant from the original set, hense, one state from the original set is needed to set the correct constant
            ## it is convienent to choose the Ref_state for this purpose.
            u_kln_P = numpy.zeros([nstates,Nparm**2 + 1,energies.itermax]) #Account for the reference state
            #Fill in the reference state
            u_kln_P[:,0,:] = energies.u_kln[:,Ref_state,:]
            #Save data files
            if not (os.path.isfile('esq_%s/ns%iNp%iQ%i.npz' % (spacename, nstates, Nparm, iq)) and savedata) or timekln: #nand gate + timing flag
                for iepsi in xrange(Nparm):
                    epsi = epsi_range[iepsi]
                    #Create Sub matrix
                    for isig in xrange(Nparm):
                        sig = sig_range[isig]
                        lndx = isig + (iepsi*Nparm)
                        #Offset by the refference state 
                        u_kln_P[:,lndx+1,:] = flamC12sqrt(epsi,sig)*energies.const_R_matrix + flamC6sqrt(epsi,sig)*energies.const_A_matrix + flamC1(q)*energies.const_q_matrix + flamC1(q)**2*energies.const_q2_matrix + energies.const_unaffected_matrix
                if not timekln:
                    #Get free energies relative to the reference state (the index l=0)
                    (DeltaF_ij, dDeltaF_ij) = mbar.computePerturbedFreeEnergies(u_kln_P, uncertainty_method='svd-ew-kab')
                if savedata and not timekln:
                    if not os.path.isdir('esq_%s' % spacename):
                        os.makedirs('esq_%s' % spacename) #Create folder
                    savez('esq_%s/ns%iNp%iQ%i.npz' % (spacename, nstates, Nparm, iq), DeltaF_ij=DeltaF_ij, dDeltaF_ij=dDeltaF_ij) #Save file
            else:
                DeltaF_file = numpy.load('esq_%s/ns%iNp%iQ%i.npz' % (spacename, nstates, Nparm, iq))
                DeltaF_ij = DeltaF_file['DeltaF_ij']
                dDeltaF_ij = DeltaF_file['dDeltaF_ij']
            #printFreeEnergy(DeltaF_ij, dDeltaF_ij)
            if not timekln:
                for iepsi in xrange(Nparm):
                    #Unwrap the data, reference state is in state 0
                    DelF[iq, iepsi,:] = DeltaF_ij[0, 1 + iepsi*Nparm:1 + (iepsi+1)*Nparm]
                    dDelF[iq, iepsi,:] = dDeltaF_ij[0, 1 + iepsi*Nparm:1 + (iepsi+1)*Nparm]
            laptime = time.clock()
            # Show timing statistics. copied from Repex.py, copywrite John Chodera
            final_time = time.time()
            elapsed_time = final_time - initial_time
            estimated_time_remaining = (final_time - run_start_time) / (iteration) * (number_of_iterations - iteration)
            estimated_total_time = (final_time - run_start_time) / (iteration) * (number_of_iterations)
            estimated_finish_time = final_time + estimated_time_remaining
            print "Iteration took %.3f s." % elapsed_time
            print "Estimated completion in %s, at %s (consuming total wall clock time %s)." % (str(datetime.timedelta(seconds=estimated_time_remaining)), time.ctime(estimated_finish_time), str(datetime.timedelta(seconds=estimated_total_time)))
            if not timekln and savedata:
                #Save a copy of just the Nparm**3 matrix in case you dont want to save a large number of files
                savez('esq_freeEnergies%s.npz'%spacename, free_energy=DelF, dfree_energy=dDelF)
        if timekln:
            pdb.set_trace()
    else: #Load the files instead
        if os.path.isfile('esq_%s/ns%iNp%iQ%i.npz' % (spacename, nstates, Nparm, Nparm-1)) and savedata: #Pull data from 
            DelF = numpy.zeros([Nparm, Nparm, Nparm])
            dDelF = numpy.zeros([Nparm, Nparm, Nparm])
            for iq in xrange(Nparm):
                sys.stdout.flush()
                sys.stdout.write('\rSave data detected, loading file %d/%d...' % (iq,Nparm-1))
                DeltaF_file = numpy.load('esq_%s/ns%iNp%iQ%i.npz' % (spacename, nstates, Nparm, iq))
                DeltaF_ij = DeltaF_file['DeltaF_ij']
                dDeltaF_ij = DeltaF_file['dDeltaF_ij']
                for iepsi in xrange(Nparm):
                    #Unwrap the datam reference state is in sate 0
                    DelF[iq, iepsi,:] = DeltaF_ij[0, 1 + iepsi*Nparm:1 + (iepsi+1)*Nparm]
                    dDelF[iq, iepsi,:] = dDeltaF_ij[0, 1 + iepsi*Nparm:1 + (iepsi+1)*Nparm]
            sys.stdout.write('\n')
        else:
            figdata = numpy.load('esq_freeEnergies%s.npz'%spacename)
            DelF = figdata['free_energy']
            dDelF = figdata['dfree_energy']
    ###############################################
    ######### END FREE ENERGY CALCULATIONS ########
    ###############################################
    #Set up basic labels for graphs
    ylabel = r'$\epsilon$ in kcal/mol'
    if sig_factor != 3:
        xlabel = r'$\sigma$ in nm'
    else:
        xlabel = r'$\sigma^{%s}$ in nm$^{%s}$' % (sig_factor, sig_factor)
        
    #Assign units
    DelF *= kjpermolTokcal/kjpermolTokT
    dDelF *= kjpermolTokcal/kjpermolTokT
    #Relative error, optional third graph
    reldDelF = numpy.abs(dDelF/DelF)
 
    #Optional Block
    ##Charging free energy by Born Model
    #tip3p_dielectric = 103
    #BDelG = numpy.zeros([Nparm, Nparm, Nparm])
    ##add in free energy of the uncharged LJ sphere of equivalent size
    #LJBDelG = numpy.zeros(BDelG.shape)
    #electron_charge = 1.60218E-19 * units.coulombs
    #dielec_vac = 8.85419E-12 * (units.coulombs)**2 / (units.joules * units.meter * units.AVOGADRO_CONSTANT_NA)
    #enot = dielec_vac / electron_charge**2 
    #e0 = enot * units.kilocalories_per_mole * units.nanometer
    #rmins = numpy.load('LJEffHS.npz')['rmins']
    #for iq in xrange(Nparm):
    #    for isig in xrange(Nparm):
    #        #BDelG[iq, :, isig] = [bornG(q_range[iq], sig_range[isig], e0, tip3p_dielectric)] * Nparm
    #        BDelG[iq, :, isig] = bornG(q_range[iq], rmins[:,isig], e0, tip3p_dielectric)
    #    LJBDelG[iq,:,:] = BDelG[iq,:,:]
    #numpy.save('BornGwithLJrdf0.npy', LJBDelG)
    #pdb.set_trace()
    
    #Optional Block
    ##Find parameters close to target values
    #ion_names = ['Li+', 'Na+', 'K+', 'Rb+', 'Cs+', 'F-', 'Cl-', 'Br-', 'I-']
    #ion_N = len(ion_names)
    #ion_F = numpy.array([-113.8, -88.7, -71.2, -66.0, -60.5, -119.7, -89.1, -82.7, -74.3])
    #ion_q = [1,1,1,1,1,-1,-1,-1,-1]
    #ionDelF = {}
    ##Interoplate to find closest value at +-1
    #ionDelF[1] = (DelF[38,:,:] + DelF[37,:,:])/2
    #ionDelF[-1] = (DelF[12,:,:] + DelF[13,:,:])/2
    #pdb.set_trace()
    #for i in xrange(ion_N):
    #    absarr = (numpy.abs(ionDelF[ion_q[i]] - ion_F[i]))
    #    idxmin = numpy.unravel_index((numpy.abs(ionDelF[ion_q[i]] - ion_F[i])).argmin(), ionDelF[ion_q[i]].shape)
    #    print "For %s with measured F=%f, the closest parameter is (%f,%f) for epsilon sigma with F=%f" %(ion_names[i], epsi_range[idxmin[0]]*kjpermolTokcal, sig_range[idxmin[1]], ionDelF[ion_q[i]][idxmin])
    #pdb.set_trace()

    ################################################
    ################ region ID #####################
    ################################################
    '''
    This section will be used to identify where extra sampling should be done. optional flag set at the start of this section
    '''
    #Does not try to rebuild the regions if its been done already, saves time when you are just wanting to generate graphs for example
    #Also avoids generating NEW points that may interfere with already sampled points if you have the sampling randomization turned on from the options
    if id_regions and not os.path.isfile('resamp_points_n%i.npy'%nstates):
        err_threshold = 0.5 #kcal/mol
        #Define the features of the array by assigning labels
        features = id_features(dDelF, err_threshold)
        if idmethod is 'lloyd':
            #The Lloyd method works, although is not as reliable in sampling phase space as dbscan, so dbscan is better developed
            #Create features based on adjacentcy
            test_struct = numpy.ones([3,3,3])
            feature_labels, num_features = ndimage.measurements.label(features)
            test_feature_labels, test_num_features = ndimage.measurements.label(features, structure=test_struct)
            """
            Important note:
            Labels of 0 in the feature_label arrays are not actually features! they are the background, so all looping will need to be done over the other indices
            """
            coms = numpy.zeros([num_features,3]) #Create the center of mass arrays
            maxes = numpy.zeros(coms.shape)
            maxes_esq = numpy.zeros(coms.shape)
            coms_esq = numpy.zeros(coms.shape) # q, epsi, and sig com
            test_coms = numpy.zeros([num_features,3]) #Create the center of mass arrays
            test_coms_esq = numpy.zeros(coms.shape) # q, epsi, and sig com
            for i in range(num_features):
               index = i + 1 #convert my counter to the feature index
               coms[i,:] = ndimage.measurements.center_of_mass(dDelF, feature_labels, index=index) #compute center of mass for each 
               test_coms[i,:] = ndimage.measurements.center_of_mass(dDelF, test_feature_labels, index=index) #compute center of mass for each 
               maxes[i,:] = ndimage.measurements.maximum_position(dDelF, feature_labels, index=index)
               #Compute the corrisponding q, epsi, and sig from each com
               fraction_along = coms[i,:] / (Nparm-1)
               test_fraction_along = test_coms[i,:] / (Nparm-1)
               coms_esq[i,0] = qStartSpace + (qEndSpace-qStartSpace)*fraction_along[0]
               coms_esq[i,1] = epsiStartSpace + (epsiEndSpace-epsiStartSpace)*fraction_along[1]
               coms_esq[i,2] = (sigStartSpace**3 + (sigEndSpace**3-sigStartSpace**3)*fraction_along[2])**(1.0/3)
               test_coms_esq[i,0] = qStartSpace + (qEndSpace-qStartSpace)*test_fraction_along[0]
               test_coms_esq[i,1] = epsiStartSpace + (epsiEndSpace-epsiStartSpace)*test_fraction_along[1]
               test_coms_esq[i,2] = sigStartSpace + (sigEndSpace-sigStartSpace)*test_fraction_along[2]
               maxes_esq[i,0] = q_range[maxes[i,0]]
               maxes_esq[i,1] = epsi_range[maxes[i,1]]
               maxes_esq[i,2] = sig_range[maxes[i,2]]
            print "Center of the %i regions with errors larger than %f kcal/mol" % (num_features, err_threshold)
            print "in units of  q, kJ/mol, and nm"
            print "Charge -- Epsilon -- Sigma"
            print coms_esq
            print "With test Structure"
            print test_coms_esq
            #Determine size of each feature to figure out which should have more
            resample_tol = 0.15
            #Use a reample tolerance of .3 instead, seems to converge faster
            #Convert to broader test structure
            num_features = test_num_features
            feature_labels = test_feature_labels
            resample_tol = 0.30
            #end modification
            Nresample = numpy.zeros(num_features, dtype=numpy.int32)
            Nsize = numpy.zeros(num_features, dtype=numpy.int32)
            for i in range(num_features):
                index = i + 1 #Convert to index
                Nsize[i] = numpy.where(feature_labels == index)[0].shape[0]
            #Points will be distributed with the following rules:
            # The pecent of points from each feature will be identified
            # There will be a finite number of points to add per distribution (10)
            # Pecrent of points in each feature will be converted to decipercent 34% = 3.4 deci%
            # Starting with the largest cluster (highest %) then in decending order, alternate celing and floor with deci%
            # The celing/floor will be the # of points distributed to that cluster
            # Points will be distriubed until we run out!
            ndistrib = 10 # Number of points to distribute
            percentSize = Nsize/float(Nsize.sum()) #Generate percentages
            deciPercents = percentSize * 10 #Convert to deciPercent (e.g. 0.34 = 34% = 3.4 deci%)
            sortedNdxMaxMin = numpy.argsort(deciPercents)[::-1] #Figure out which has the max
            updown = numpy.ceil
            pointsleft = ndistrib
            for index in sortedNdxMaxMin:
                pointsToGive = updown(deciPercents[index])
                if pointsToGive <= pointsleft:
                    Nresample[index] = pointsToGive
                    pointsleft -= pointsToGive
                elif pointsToGive > pointsleft and pointsleft != 0:
                    Nresample[index] = pointsleft
                    pointsleft -= pointsleft
                else:
                    Nresample[index] = 0
                if updown is numpy.ceil:
                    updown = numpy.floor
                else:
                   updown = numpy.ceil 
            #Nresample[Nsize/float(Nsize.sum()) > resample_tol] = 3 #Resample at > 30% of the total points
            #Nresamp_total = Nresample.sum()
            resamp_points = numpy.zeros([Nresample.sum(), 3])
            closest_interiors = numpy.zeros(resamp_points.shape)
            #Tesalate over where multiple samples are needed based on k-clustering Lloyd's algorithm
            #pdb.set_trace()
            resamp_counter = 0
            for i in xrange(num_features):
                index = i + 1 #Convert to index
                #if Nresample[i] > 1:
                if Nresample[i] > 0:
                    feature_indices = numpy.transpose(numpy.array(numpy.where(feature_labels==index))) #Creates a NxD matrix where N=feature.size
                    #feature_indices = numpy.where(feature_labels==i)
                    mu, clusters = find_centers(feature_indices, Nresample[i], dDelF)
                    for n in range(Nresample[i]):
                        fraction_along = mu[n] / (Nparm-1)
                        resamp_points[resamp_counter,0] = qStartSpace + (qEndSpace-qStartSpace)*fraction_along[0]
                        resamp_points[resamp_counter,1] = epsiStartSpace + (epsiEndSpace-epsiStartSpace)*fraction_along[1]
                        resamp_points[resamp_counter,2] = (sigStartSpace**3 + (sigEndSpace**3-sigStartSpace**3)*fraction_along[2])**(1.0/3)
                        resamp_counter += 1
                        #closest_interiors = closest_index(mu[n], feature_labels, i)
                #else:
                    #Comment out to ignore these for weaker features
                    #resamp_points[resamp_counter,:] = coms_esq[i,:]
                    #resamp_counter += 1
        elif idmethod is 'dbscan':
            clusters_found = False #Flag to ensure we have clusters.
            while not clusters_found:
                #Set the common vertex, the reference state
                vertices = numpy.zeros([1,3])
                vertices[0,0] = q_samp_space[Ref_state]
                vertices[0,1] = epsi_samp_space[Ref_state]
                vertices[0,2] = sig_samp_space[Ref_state]
                #Initilize density based scanner
                scanner = dbscan.dbscan(features, dDelF)
                #Find features
                feature_labels, num_features = scanner.generate_neighborhoods()
                vertex_index = [] #Store which cluster labels to make a vertex out of
                fsize = numpy.zeros(num_features,dtype=int)
                #Tabulate Features
                for i in xrange(num_features):
                    index = i + 1
                    #This excludes the 0 index (background)
                    fsize[i] = numpy.where(feature_labels == index)[0].size
                #Find features we care about
                for i in xrange(num_features):
                    if fsize[i]/float(fsize.sum()) >= 0.1: #Feature larger than 10% of all non-background features
                        vertex_index.append(i+1) #Reaccount for excluding the 0 index
                if len(vertex_index) == 0: #Check for only a whole bunch of small clusters
                    maxV = 3
                    nV = 0
                    for i in numpy.argsort(fsize)[::-1]: #Sort sizes from max to smallest
                        try:
                            #Add next largest cluster if possible
                            vertex_index.append(i+1)
                            nV += 1
                            if nV >= maxV:
                                clusters_found = True
                                break
                        except: #Exception for no clusters available, just pass since no clusters will have been found
                            #This should only happen when fsize is 0 or something.
                            pass
                else:
                    #Clusters have been found, continue algorithm
                    clusters_found = True
                #If no small clusters are found either, lower the error threshold until 1/3 of the grid points are above the threshold
                if len(vertex_index) == 0:
                    while float(numpy.where(dDelF > err_threshold)[0].size) / dDelF.size  < 1.0/3:
                        err_threshold -= 0.001 #Arbitrary reduction, all we really care about is creating regions of uncertainty again
                    features = id_features(dDelF, err_threshold)
                    #Write the new error threhold for documentation
                    numpy.save('err_threshold_n%i.npy' % nstates, numpy.array(err_threshold))
            #Create master vertex system for the graph
            Nnew_vertices = len(vertex_index)
            if db_rand:
                #Randomly choose a point inside each region
                new_points = numpy.zeros([Nnew_vertices,3])
                new_points_esq = numpy.zeros(new_points.shape)
                #Get the slices of each region
                shapes = ndimage.find_objects(feature_labels)
                for i in range(Nnew_vertices):
                    index = vertex_index[i]
                    shape_slices = shapes[index-1] # The features are in a list with the list index = feature # - 1
                    pointfound = False #Ensure the new random point is inside the slices for the cluster
                    while not pointfound:
                        ndx_point = numpy.zeros(3)
                        #Roll random numbers
                        rng = numpy.random.rand(3)
                        #assign an index based on top bottom
                        for dim in range(3):
                            start = int(shape_slices[dim].start)
                            end = int(shape_slices[dim].stop) - 1 #Account for the fact that the end index is 1 larger than the available index's
                            delta = end-start
                            ndx_point[dim] = start + rng[dim]*delta
                        #See what the closest index is
                        rndx, near_ndx = closest_point_with_index(ndx_point, feature_labels)
                        if near_ndx == index:
                            pointfound = True
                            new_points[i,:] = ndx_point
                    fraction_along = new_points[i,:] / (Nparm-1)
                    new_points_esq[i,0] = qStartSpace + (qEndSpace-qStartSpace)*fraction_along[0]
                    new_points_esq[i,1] = epsiStartSpace + (epsiEndSpace-epsiStartSpace)*fraction_along[1]
                    new_points_esq[i,2] = (sigStartSpace**3 + (sigEndSpace**3-sigStartSpace**3)*fraction_along[2])**(1.0/3)
                vertices = numpy.concatenate((vertices,new_points_esq))
                resamp_points = new_points_esq
            else:
                #Vertex is the center of mass of the region
                coms = numpy.zeros([Nnew_vertices,3]) #Create the center of mass arrays
                coms_esq = numpy.zeros(coms.shape) # q, epsi, and sig com
                #Trap nan's
                nandDelF = dDelF.copy()
                nandDelF[numpy.isnan(dDelF)] = numpy.nanmax(dDelF)
                for i in range(Nnew_vertices):
                    index = vertex_index[i] #convert my counter to the feature index
                    coms[i,:] = ndimage.measurements.center_of_mass(nandDelF, feature_labels, index=index) #compute center of mass for each 
                    #Compute the corrisponding q, epsi, and sig from each com
                    fraction_along = coms[i,:] / (Nparm-1)
                    coms_esq[i,0] = qStartSpace + (qEndSpace-qStartSpace)*fraction_along[0]
                    coms_esq[i,1] = epsiStartSpace + (epsiEndSpace-epsiStartSpace)*fraction_along[1]
                    coms_esq[i,2] = (sigStartSpace**3 + (sigEndSpace**3-sigStartSpace**3)*fraction_along[2])**(1.0/3)
                vertices = numpy.concatenate((vertices,coms_esq))
                resamp_points = coms_esq
            numpy.save('vertices{0:d}.npy'.format(nstates), vertices) #Write the vertices to disk
            #Generate the complete connectivity network (graph) in upper triangle matrix for sparse processing
            nv = vertices.shape[0]
            lengths = numpy.zeros([nv,nv])
            #Convert to index lengths on vertices before measuring lengths, this is the only places its needed
            ndx_vertices = numpy.zeros(vertices.shape)
            qSE = {'start':qStartSpace,'end':qEndSpace}
            eSE = {'start':epsiStartSpace,'end':epsiEndSpace}
            sSE = {'start':sigStartSpace,'end':sigEndSpace,'factor':3}
            SEall = [qSE, eSE, sSE]
            for dim in range(3):
                ndx_vertices[:,dim] = esq_to_ndx(vertices[:,dim], **SEall[dim])
            for v in xrange(nv):
                for vj in xrange(v,nv):
                    lengths[v,vj] = numpy.linalg.norm(ndx_vertices[v,:]-ndx_vertices[vj,:])
            #Compute the minium spanning tree
            sparse_mst = csgraph.minimum_spanning_tree(lengths)
            #Convert to human-readable format
            mst = sparse_mst.toarray()
            #Generate the resample points, starting with the new vertices
            nline = 51 #Number of points to uniformly place on each edge
            line_frac = linspace(0,1,nline)
            edgen = 0
            for v in xrange(nv):
                for vj in xrange(v,nv):
                    #Check if there is an Edge connecting the vertices
                    if mst[v,vj] > 0:
                        #Generate the Edge points
                        edge = numpy.zeros([nline,3])
                        edgendx = numpy.zeros([nline,3])
                        edgedelF = numpy.zeros([nline])
                        #Edge is somewhat directional, 
                        #but since we only care about "border" where the largest transition occurs
                        #This problem resolves itself
                        edge[:,0] = vertices[v,0] + (vertices[vj,0] - vertices[v,0])*line_frac
                        edgendx[:,0] = esq_to_ndx(edge[:,0], qStartSpace, qEndSpace)
                        edge[:,1] = vertices[v,1] + (vertices[vj,1] - vertices[v,1])*line_frac
                        edgendx[:,1] = esq_to_ndx(edge[:,1], epsiStartSpace, epsiEndSpace)
                        edge[:,2] = (vertices[v,2]**3 + (vertices[vj,2]**3 - vertices[v,2]**3)*line_frac)**(1.0/3)
                        edgendx[:,2] = esq_to_ndx(edge[:,2], sigStartSpace, sigEndSpace, factor=3)
                        #Determine the average "error" of each point based on cubic interpolation to the 8 cube points around it
                        for point in xrange(nline):
                            edgedelF[point] = cubemean(edgendx[point,:],dDelF)
                        #Generate a sobel filter to find where the sudden change in edge is
                        laplaceline = scipy.ndimage.filters.sobel(edgedelF)
                        #Alternate (original): use a laplace boundary detection. No real difference between the two methods
                        #laplaceline = scipy.ndimage.filters.laplace(edgedelF)
                        ####### Graph the dDelF for each edge, optional debugging block ########
                        #f,(b,a) = plt.subplots(1,2)
                        #a.plot(line_frac, laplaceline)
                        #b.plot(line_frac, edgedelF)
                        #a.set_title('Edge Detection\nTransform')
                        #b.set_title(r'$\delta\Delta F$')
                        #f.savefig('sorbel_edge{0:d}'.format(edgen), bbox_inches='tight')
                        #edgen+=1
                        #plt.show()
                        ###### END DEBUG BLOCK #####
                        #Find the point where this change is the largest and add it to the resampled points
                        if db_rand and False: #Disabled for now
                            #Find the point in the edge which has the maximum distance from its closest point
                            #i.e. the closest point is further away than any other point's neighbors
                            mindata = {'ndx':-1, 'mindist':0}
                            for i in xrange(nline): #Go through each point in the edge
                                #Set the slice array
                                noti = numpy.ones(nline, dtype=bool)
                                noti[i] = 0
                                distset = numpy.abs(edgedelF[noti] - edgedelF[i]) #Take the distance btween every other point
                                if distset.min() > mindata['mindist'] and edgedelF[i] > err_threshold : #If a new maximum minimum-distance is found, set it (must also be a region of uncertainty)
                                    mindata['mindist'] = distset.min()
                                    mindata['ndx'] = i
                            #Assign the new point to be sampled
                            new_point = edge[mindata['ndx'],:]
                        else:
                            boundaryline = int(numpy.nanargmax(numpy.abs(laplaceline)))
                            new_point = edge[boundaryline,:]
                        #Check to make sure its not in our points already (sometimes happens with MST)
                        #Added spaces to help human-readability
                        if not numpy.any([   numpy.allclose(numpy.array([q_samp_space[i],epsi_samp_space[i],sig_samp_space[i]]), new_point)   for i in xrange(nstates)]):
                            #Cast new_point to the correct dims before appending
                            resamp_points = numpy.concatenate((resamp_points, numpy.array([new_point])))
        numpy.savetxt('resamp_points_n%i.txt'%nstates, resamp_points)
        numpy.save('resamp_points_n%i.npy'%nstates, resamp_points)

    ################################################
    ################# PLOTTING #####################
    ################################################
    #Plot the sigma and epsilon free energies
    if relativeErr:
        f,(Fplot,dFplot,rdFplot) = plt.subplots(3,1,sharex=True)
        rdFig = f
        plotlist=[Fplot,dFplot,rdFplot]
    else:
        f,(Fplot,dFplot) = plt.subplots(2,1,sharex=True)
        g,rdFplot = plt.subplots(1,1)
        rdFig = g
        plotlist=[Fplot,dFplot]
    '''
    Observant readers will notice that DelF and dDelF are in dimensions of [epsi,sig] but I plot in sig,epsi. That said, the color map is CORRECT with this method... somehow. I questioned it once and then decided to come back to it at a later date.
    '''
    import matplotlib.animation as ani
    cvmax = DelF.max()*1.01
    cvmin = DelF.min()*1.01
    #Set the default error tolerance
    try:
        errorlimits = numpy.load('n24_error_lims.npy')
        cdvmin = errorlimits[0]
        cdvmax = errorlimits[1]
    except:
        cdvmin = dDelF.min()*1.01
        cdvmax = dDelF.max()*1.01
        if nstates == 24:
            numpy.save('n24_error_lims.npy', numpy.array([dDelF.min()*1.01, dDelF.max()*1.01]))
    try:
        relerrlims = numpy.load('max_rel_err_lims.npy')
        crdvmin = relerrlims[0]
        crdvmax = relerrlims[1]
        #curmax = reldDelF.max()*1.01
        curmax = reldDelF.mean() + 2*numpy.sqrt(reldDelF.var())
        if curmax > crdvmax:
            print 'Max relative error exceeded, extending maximum to %f' % curmax
            numpy.save('max_rel_err_lims.npy', numpy.array([0,curmax]))
    except:
        crdvmin = 0
        #crdvmax = reldDelF.max()*1.01
        crdvmax = reldDelF.mean() + 2*numpy.sqrt(reldDelF.var())
        numpy.save('max_rel_err_lims.npy', numpy.array([0,crdvmax]))
    #imgFplot = Fplot.pcolormesh(sig_range**sig_factor,epsi_range,DelF[(Nparm-1)/2,:,:], vmax=cvmax, vmin=cvmin)
    imgFplot = Fplot.pcolormesh(sig_range**sig_factor,epsi_range,DelF[(Nparm-1)/2,:,:])
    #imgFplot = Fplot.pcolormesh(sig_range**sig_factor,epsi_range,[])
    #Set the colorbar
    divFplot = mal(Fplot)
    caxFplot = divFplot.append_axes('right', size='5%', pad=0.05)
    cFplot = f.colorbar(imgFplot, cax=caxFplot)
    #cFplot.set_clim(vmin=cvmin, vmax=cvmax)
    #set the minmax colorscales
    #print cFplot.get_clim()
    #cvmin, cvmax = (-21.971123537881027, 20.78176716595965) #These are the 11 nstate plots
    #cvmin, cvmax = (-14.542572154421956, 8.6595207877425739)
    ####### Error plot #######
    #imgdFplot = dFplot.pcolormesh(sig_range**sig_factor,epsi_range,dDelF[(Nparm-1)/2,:,:], vmax=cdvmax, vmin=cdvmin)
    imgdFplot = dFplot.pcolormesh(sig_range**sig_factor,epsi_range,dDelF[(Nparm-1)/2,:,:])
    #imgdFplot = dFplot.pcolormesh(sig_range**sig_factor,epsi_range,dDelF[5,:,:])
    #imgdFplot = dFplot.pcolormesh(sig_range**sig_factor,epsi_range,[])
    divdFplot = mal(dFplot)
    caxdFplot = divdFplot.append_axes('right', size='5%', pad=0.05)
    #Set the minmax colorscales
    #print imgdFplot.get_clim()
    #cdvmin, cdvmax = (0.00019094581786378227, 0.45022226894935008) #These are the 11 nstate plots
    #cdvmin, cdvmax = (3.1897634261829015e-05, 0.22292838017499619) 
    imgdFplot.set_clim(vmin=cdvmin, vmax=cdvmax)
    cdFplot = f.colorbar(imgdFplot, cax=caxdFplot)
    ####### Relative Error Plot ########
    imgrdFplot = rdFplot.pcolormesh(sig_range**sig_factor,epsi_range,reldDelF[(Nparm-1)/2,:,:])
    divrdFplot = mal(rdFplot)
    caxrdFplot = divrdFplot.append_axes('right', size='5%', pad=0.05)
    imgrdFplot.set_clim(vmin=crdvmin, vmax=crdvmax)
    crdFplot = rdFig.colorbar(imgrdFplot, cax=caxrdFplot)

    sup_title_template = r'$\Delta G$ (top) and $\delta\Delta G$(bottom) with $q=%.2f$ for LJ Spheres' + '\n in units of kcal/mol'
    ftitle = f.suptitle('')
    #Set up the empty plots
    Fline, = Fplot.plot([], [], linewidth=2, color='k')
    dFline, = dFplot.plot([], [], linewidth=2, color='w')
    F_scatter_noref, = Fplot.plot([], [], linestyle='', markersize=5, color='k', marker='x', markeredgewidth=2)
    dF_scatter_noref, = dFplot.plot([], [], linestyle='', markersize=5, color='w', marker='x', markeredgewidth=2)
    F_scatter_ref, = Fplot.plot([], [], linestyle='', markersize=6, color='k', marker='D', markeredgewidth=2)
    dF_scatter_ref, = dFplot.plot([], [], linestyle='', markersize=6, color='w', marker='D', markeredgewidth=2, markeredgecolor='w')
    #Create the scatter sampled data
    for ax in plotlist:
        ax.set_yscale(spacename)
        ax.set_xscale(spacename)
        ax.set_ylim([epsiPlotStart,epsiPlotEnd])
        ax.set_xlim([sigPlotStart,sigPlotEnd])
        ax.patch.set_color('grey')
    f.subplots_adjust(hspace=0.02)
    f.text(0.05, .5, ylabel, rotation='vertical', horizontalalignment='center', verticalalignment='center', fontsize=20)
    if relativeErr:
        rdFplot.set_xlabel(xlabel, fontsize=20)
    else:
        dFplot.set_xlabel(xlabel, fontsize=20)
    #Animate the figures
    q_ref = q_samp_space[Ref_state]
    sig_ref = sig_samp_space[Ref_state]
    epsi_ref = epsi_samp_space[Ref_state]
    def cleanup():
        imgFplot.set_array([])
        imgdFplot.set_array([])
        imgrdFplot.set_array([])
        ftitle.set_text('')
        F_scatter_ref.set_data([], [])
        dF_scatter_ref.set_data([], [])
        F_scatter_noref.set_data([], [])
        dF_scatter_noref.set_data([], [])
        Fline.set_data([], [])
        dFline.set_data([], [])
        #for iscatter,discatter in zip(Fscatters,dFscatters):
        #    iscatter.set_data([],[])
        #    discatter.set_data([],[])
        if relativeErr:
            return imgFplot, imgdFplot, imgrdFplot, ftitle, F_scatter_noref, dF_scatter_noref, F_scatter_ref, dF_scatter_ref, Fline, dFline
        else:
            return imgFplot, imgdFplot, ftitle, F_scatter_noref, dF_scatter_noref, F_scatter_ref, dF_scatter_ref, Fline, dFline
    def moveq(qndx):
        q = q_range[qndx]
        #I have to create a secondary pcolormesh set since pcolormesh truncates the array size to make the render faster (I dont think pcolor requires this, would but would be slower)
        #If you don't do this, then it creates a weird set of lines which make no sense
        scrapFig, (scrapF, scrapdF, scraprdF) = plt.subplots(3,1)
        scrapFplot = scrapF.pcolormesh(sig_range**sig_factor,epsi_range,DelF[qndx,:,:])
        #Lock down the color choice for the error plot
        scrapdFplot = scrapdF.pcolormesh(sig_range**sig_factor,epsi_range,dDelF[qndx,:,:], vmax=cdvmax, vmin=cdvmin)
        #Lock down the color choice for the relative error plot
        scraprdFplot = scraprdF.pcolormesh(sig_range**sig_factor,epsi_range,reldDelF[qndx,:,:],vmax=crdvmax, vmin=crdvmin)
        #Reassign the plots, if you did not use an already generated array, you would need to .ravel() on the array you feed to set_array()
        imgFplot.set_array(scrapFplot.get_array())
        imgdFplot.set_array(scrapdFplot.get_array())
        imgrdFplot.set_array(scraprdFplot.get_array())
        ftitle.set_text(sup_title_template % q)
        Dmax = DelF[qndx,:,:].max()
        Dmin = DelF[qndx,:,:].min()
        dDmax = dDelF[qndx,:,:].max()
        dDmin = dDelF[qndx,:,:].min()
        imgFplot.set_clim(vmin=Dmin, vmax=Dmax)
        #imgdFplot.set_clim(vmin=dDmin, vmax=dDmax)
        #Set up the scatters
        #Set the Q scatters correctly
        qsampled = numpy.where(q_samp_space == q) #Get all sampled states from the current q value
        epsi_qsamp = epsi_samp_space[qsampled]
        sig_qsamp = sig_samp_space[qsampled]
        F_scatter_noref.set_data(sig_qsamp**sig_factor, epsi_qsamp)
        dF_scatter_noref.set_data(sig_qsamp**sig_factor, epsi_qsamp)
        if numpy.any(Ref_state == qsampled[0]):
            F_scatter_ref.set_data(sig_ref**sig_factor, epsi_ref)
            dF_scatter_ref.set_data(sig_ref**sig_factor, epsi_ref)
        else:
            #F_scatter_ref.set_data([], [])
            #dF_scatter_ref.set_data([], [])
            F_scatter_ref.set_data(sig_ref**sig_factor, epsi_ref)
            dF_scatter_ref.set_data(sig_ref**sig_factor, epsi_ref)
        if q == 0:
            Fline.set_data(sig_range**sig_factor, epsi_plot_range)
            dFline.set_data(sig_range**sig_factor, epsi_plot_range)
        else:
            Fline.set_data([], [])
            dFline.set_data([], [])
        #Cleanup scrap figure to avoid memory buildup
        plt.close(scrapFig)
        if relativeErr:
            return imgFplot, imgdFplot, imgrdFplot, ftitle, F_scatter_noref, dF_scatter_noref, F_scatter_ref, dF_scatter_ref, Fline, dFline
        else:
            return imgFplot, imgdFplot, ftitle, F_scatter_noref, dF_scatter_noref, F_scatter_ref, dF_scatter_ref, Fline, dFline
    aniU = ani.FuncAnimation(f, moveq, range(Nparm), interval=150, blit=False, init_func=cleanup)
    if relativeErr:
        filename='Animated_charging_rel{myint:{width}}.mp4'.format(myint=nstates, width=len(str(nstates)))
    else:
        filename='Animated_charging{myint:{width}}.mp4'.format(myint=nstates, width=len(str(nstates)))
    #pdb.set_trace()
    aniU.save(filename, dpi=400)
    #save a single frame
    #qframe=40
    #moveq(qframe)
    #f.savefig('DelF_Nstate_%i_Qndx_%i.png' % (nstates, qframe), bbox_inches='tight', dpi=400)
    #if savefigs:
    #    f.patch.set_alpha(0.0)
    #    #f.savefig('LJ_GdG_ns%i_es%i_N%i_em%1.1f.png' % (nstates, sig_factor, Nparm, epsiEndSpace), bbox_inches='tight', dpi=600)  
    #    print "Making the PDF, boss!"
    #    #f.savefig('LJ_GdG_ns%i_es%i_N%i_em%1.1f.pdf' % (nstates, sig_factor, Nparm, epsiEndSpace), bbox_inches='tight')  
    #    #f.savefig('LJ_GdG_ns%i_es%i_N%i_em%1.1f.eps' % (nstates, sig_factor, Nparm, epsiEndSpace), bbox_inches='tight')  
    #else:
    #    plt.show()
####################################################################################
####################################################################################
####################################################################################

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--nstates", dest="nstates", default=None, help="Set the number of states", metavar="NSTATES")
    (options, args) = parser.parse_args()
    x = numpy.load('qes.npy')
    qs = x[:,0]
    es = x[:,1]
    ss = x[:,2]
    if options.nstates is None:
        nstates = len(qs)
    else:
        nstates = int(options.nstates)
    #If this script is run with manaual execution, break here since its probably being run as debugging anyways.
    pdb.set_trace()
    execute(nstates, qs, es, ss)
