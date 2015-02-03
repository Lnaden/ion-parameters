from basisanalyze import *
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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
import time
import datetime
import sys
from mpl_toolkits.mplot3d import axes3d
import timeseries
import dbscan
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph

"""
Special module to constuct H and S from set data
"""

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
T = 298 * units.kelvin
kT = kB*T
kjpermolTokT = units.kilojoules_per_mole / kT
kjpermolTokcal = 1/4.184
graphsfromfile=True #Generate graphs from the .npz arrays
savedata = True #Save dhdl data
masked=False
load_ukln = True
timekln = False #Timing information
#subsample_method = 'per-state'
#subsample_method = 'all'
subsample_method = 'presubsampled'

#logspace or linspace
#narg = len(sys.argv)
#if narg > 1:
#    if sys.argv[1] == 'log':
#        spacing=logspace
#    elif sys.argv[1] == 'lin':
#        spacing=linspace
#    else:
#        spacing=linspace
#else:
#    spacing=linspace
spacing=linspace


#Main output controling vars:
Nparm = 51 #51, 101, or 151
plotReal = False
sig_factor=1
annotatefig = False
savefigs = False
if Nparm == 151 or True:
    alle = True
else:
   alle = False

#Real molecule sampling
realname = ['UAm', 'NOP', 'C60', 'LJ6', 'null', 'LJ11']
nreal = len(realname)
realepsi = numpy.array([1.2301, 3.4941, 1.0372, 0.7600, 0, 0.8])
realsig  = numpy.array([0.3730, 0.6150, 0.9452, 1.0170, 0, 0.3])
realq    = numpy.array([     0,      0,      0,      0, 0,   0])

#epsi_max = 0.40188
#sig_max = 1.31453


#Test numpy's savez_compressed formula
try:
    savez = numpy.savez_compressed
except:
    savez = numpy.savez


################ SUBROUTINES ##################
def my_annotate(ax, s, xy_arr=[], *args, **kwargs):
  ans = []
  an = ax.annotate(s, xy_arr[0], *args, **kwargs)
  ans.append(an)
  d = {}
  try:
    d['xycoords'] = kwargs['xycoords']
  except KeyError:
    pass
  try:
    d['arrowprops'] = kwargs['arrowprops']
  except KeyError:
    pass
  for xy in xy_arr[1:]:
    an = ax.annotate(s, xy, alpha=0.0, xytext=(0,0), textcoords=an, **d)
    ans.append(an)
  return ans

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
    #for dx in [numpy.ceil, numpy.floor]:
    #    for dy in [numpy.ceil, numpy.floor]:
    #        for dz in [numpy.ceil, numpy.floor]:
    #            gridpt = (dx(px), dy(py), dz(pz))
    #            grid_points.append(numpy.array(gridpt))
    #            grid_val.append(grid[gridpt])
    #netmean = scipy.interpolate.interpn(grid_points, grid_val, point)

    return c


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

#if __name__=="__main__":
def execute(nstates, q_samp_space, epsi_samp_space, sig_samp_space):
    #Initilize limts
    sig3_samp_space = sig_samp_space**3
    #These are the limits used to compute the constant matricies
    #They should match with LJ 0 and 5, ALWAYS
    q_min = -2.0
    q_max = +2.0
    #epsi_min = 0.1 #kJ/mol
    #epsi_max = 0.65
    #sig_min = 0.25
    #sig_max = 0.95796625
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
    #dhdlstart = 34 #Row where data starts
    #dhdlenergy = 1 #column where energy is
    #dhdlstates = 4 #Column where dhdl to other states starts, also coulmn for U0
    g_en_start = 19 #Row where data starts in g_energy output
    g_en_energy = 1 #Column where the energy is located
    #niterations = len(open('lj10/prod/energy10_10.xvg','r').readlines()[g_en_start:])
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
        #ukln_consts = numpy.load('esq_ukln_consts_n%i.npz'%nstates)
        #u_kln = ukln_consts['u_kln']
        #const_R_matrix = ukln_consts['const_R_matrix'] 
        #const_A_matrix = ukln_consts['const_A_matrix'] 
        #const_unaffected_matrix = ukln_consts['const_unaffected_matrix']
        #const_q_matrix = ukln_consts['const_q_matrix']
        #const_q2_matrix = ukln_consts['const_q2_matrix']
    else:
        #Initial u_kln
        energies = consts(nstates)
        g_t = numpy.zeros([nstates])
        #Read in the data
        for k in xrange(nstates):
            print "Importing LJ = %02i" % k
            energy_dic = {'full':{}, 'rep':{}}
            #Try to load the sub filenames
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
    for l in xrange(nstates):
        sanity_kln[:,l,:] = lamC12[l]*energies.const_R_matrix + lamC6[l]*energies.const_A_matrix + lamC1[l]*energies.const_q_matrix + lamC1[l]**2*energies.const_q2_matrix + energies.const_unaffected_matrix
    del_kln = numpy.abs(energies.u_kln - sanity_kln)
    print "Max Delta: %f" % numpy.nanmax(del_kln)
    rel_del_kln = numpy.abs(del_kln/energies.u_kln)
    if numpy.nanmax(del_kln) > 2:
        pdb.set_trace()
    #pdb.set_trace()
    ##################################################
    ############### END DATA INPUT ###################
    ##################################################
   #Create master uklns
    #Convert to dimless
    energies.dimless()
    #u_kln = u_kln * kjpermolTokT 
    #const_R_matrix = const_R_matrix * kjpermolTokT 
    #const_A_matrix = const_A_matrix * kjpermolTokT 
    #const_unaffected_matrix = const_unaffected_matrix * kjpermolTokT
    #const_q_matrix *= kjpermolTokT
    #const_q2_matrix *= kjpermolTokT
    includeRef = False
    if includeRef:
        offset=1
    else:
        offset=0
    Nallstates = Nparm + nstates + offset #1 to recreate the refstate each time
    sig_range = (spacing(sigStartSpace**3,sigEndSpace**3,Nparm))**(1.0/3)
    epsi_range = spacing(epsiStartSpace,epsiEndSpace,Nparm)
    q_range = spacing(qStartSpace,qEndSpace,Nparm)
    epsi_plot_range = spacing(epsiStartSpace,epsi_max,Nparm)
    #Load subsequent f_ki
    f_ki_loaded = False
    state_counter = nstates
    #pdb.set_trace()
    #start = time.clock()
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
        #f_ki=numpy.array([0. ,61.20913184 ,71.40827393 ,75.87878531 ,78.40211785 ,79.89587372 ,80.45288761 ,80.28963586 ,79.71483901 ,78.90630752 ,77.90602495 ,0.5571373 ,64.03428624 ,20.01885445 ,-58.8966979 ,-178.11292884 ,-343.48493961 ,-556.63789832 ,70.70837529 ,30.71331917 ,-40.28879673 ,-144.71442394 ,-284.20819285 ,-460.07678445 ,-210.74990763 ,-202.3625391 ,-211.89582577 ,-217.2418002 ,-168.97823733 ,-158.94266495 ,-165.72416028 ,57.79253217 ,-195.03626708 ,-214.19139447 ,-196.65374506 ,-206.69571675 ,-270.11113276 ,-408.83567163 ,-147.95744809 ,-127.26705178 ,-192.96912003 ,-202.04056754 ,-196.08529618 ,-207.33238137 ,-155.20225707 ,-156.03612919 ,-91.06462805 ,3.81078618 ,-279.65874533])
        #comp = ncdata('complex', '.', u_kln_input=u_kln, nequil=0000, save_equil_data=True, manual_subsample=True, compute_mbar=True, verbose=True, mbar_f_ki=f_ki)
        comp = ncdata('complex', '.', u_kln_input=energies.u_kln, nequil=0000, save_equil_data=True, subsample_method=subsample_method, compute_mbar=True, verbose=True, mbar_f_ki=f_ki)
        #pdb.set_trace()
    except:
        comp = ncdata('complex', '.', u_kln_input=energies.u_kln, nequil=0000, save_equil_data=True, subsample_method=subsample_method, compute_mbar=True, verbose=True)
    if not f_ki_loaded or f_ki_n != nstates:
        try:
            numpy.save_compressed('esq_f_k_{myint:{width}}.npy'.format(myint=nstates, width=len(str(nstates))), comp.mbar.f_k)
        except:
            numpy.save('esq_f_k_{myint:{width}}.npy'.format(myint=nstates, width=len(str(nstates))), comp.mbar.f_k)
    #end = time.clock()
    #print "Time passed:"
    #print end - start
    #sys.exit(0)
    #(DeltaF_ij, dDeltaF_ij) = comp.mbar.getFreeEnergyDifferences(uncertainty_method='svd-ew')
    #printFreeEnergy(DeltaF_ij,dDeltaF_ij)
    #Reun from the subsampled data
    maxN = comp.N_k.max()
    #if comp.subsample_method == 'per-state':
    #    for k in xrange(nstates):
    #        ndxs = comp.retained_indices[k, :comp.N_k[k]]
    #        Nndxs = len(ndxs)
    #        u_kln[k,:,:Nndxs] = u_kln[k,:,ndxs].T
    #        const_R_matrix[k,:Nndxs] = const_R_matrix[k,ndxs].T
    #        const_A_matrix[k,:Nndxs] = const_A_matrix[k,ndxs].T
    #        const_unaffected_matrix[k,:Nndxs] = const_unaffected_matrix[k,ndxs].T
    #        const_q_matrix[k,:Nndxs] = const_q_matrix[k,ndxs].T
    #        const_q2_matrix[k,:Nndxs] = const_q2_matrix[k,ndxs].T
    #else:
    #    u_kln = u_kln[:,:,comp.retained_indices]
    #    const_R_matrix = const_R_matrix[:,comp.retained_indices]
    #    const_A_matrix = const_A_matrix[:,comp.retained_indices]
    #    const_unaffected_matrix = const_unaffected_matrix[:,comp.retained_indices]
    #    const_q_matrix = const_q_matrix[:,comp.retained_indices]
    #    const_q2_matrix = const_q2_matrix[:,comp.retained_indices]
    #    niterations = len(comp.retained_indices)
    Ref_state = 1 #Reference state of sampling to pull from
    #pdb.set_trace()
    if not (graphsfromfile) or not (os.path.isfile('hs_%s/ns%iNp%iQ%i.npz' % (spacename, nstates, Nparm, Nparm-1)) and savedata) or timekln: #nand gate +timing flag
        #Create numpy arrys: q, epsi, sig
        DelF = numpy.zeros([Nparm, Nparm, Nparm])
        dDelF = numpy.zeros([Nparm, Nparm, Nparm])
        DelU = numpy.zeros([Nparm, Nparm, Nparm])
        dDelU = numpy.zeros([Nparm, Nparm, Nparm])
        DelS = numpy.zeros([Nparm, Nparm, Nparm])
        dDelS = numpy.zeros([Nparm, Nparm, Nparm])
        #Populate energies
        run_start_time = time.time()
        number_of_iterations = Nparm
        iteration = 0
        #for iq in xrange(Nparm):
        #!!! 0-17, 17-34, 34-Nparm
        #!!! 0-25, 25-Nparm
        number_of_iterations = Nparm
        for iq in xrange(Nparm):
            #Grab charge
            q = q_range[iq]
            initial_time = time.time()
            iteration += 1
            print "Q index: %i/%i" % (iq, Nparm-1)
            #Using PerturpedFreeEnergies instead of recreating the MBAR object every time. Saves time with same accuracy
            #Perturbed assumes all l states are unsampled
            u_kln_P = numpy.zeros([nstates,Nparm**2 + 1,maxN]) 
            #Fill in ref state
            u_kln_P[:,0,:] = energies.u_kln[:,Ref_state,:]
            #Save data files
            if not (os.path.isfile('hs_%s/ns%iNp%iQ%i.npz' % (spacename, nstates, Nparm, iq)) and savedata) or timekln: #nand gate + timing flag
                for iepsi in xrange(Nparm):
                    epsi = epsi_range[iepsi]
                    #Create Sub matrix
                    #u_kln_sub = numpy.zeros([Nallstates,Nallstates,niterations])
                    #u_kln_sub[:nstates,:nstates,:] = u_kln
                    #DEBUG: Rebuild the reference state
                    #if includeRef:
                    #    Repsi = epsi_samp_space[Ref_state]
                    #    Rsig = sig_samp_space[Ref_state]
                    #    u_kln_sub[:nstates,nstates,:] = flamC12sqrt(Repsi,Rsig)*const_R_matrix + flamC6sqrt(Repsi,Rsig)*const_A_matrix + const_unaffected_matrix
                    for isig in xrange(Nparm):
                        sig = sig_range[isig]
                        lndx = isig + (iepsi*Nparm)
                        u_kln_P[:,lndx+1,:] = flamC12sqrt(epsi,sig)*energies.const_R_matrix + flamC6sqrt(epsi,sig)*energies.const_A_matrix + flamC1(q)*energies.const_q_matrix + flamC1(q)**2*energies.const_q2_matrix + energies.const_unaffected_matrix
                if not timekln:
                    (DeltaF_ij, dDeltaF_ij, DeltaU_ij, dDeltaU_ij, DeltaS_ij, dDeltaS_ij) = comp.mbar.computePerturbedEntropyAndEnthalpy(u_kln_P, uncertainty_method='svd-ew-kab', testM='new')
                if savedata and not timekln:
                    if not os.path.isdir('hs_%s' % spacename):
                        os.makedirs('hs_%s' % spacename) #Create folder
                    savez('hs_%s/ns%iNp%iQ%i.npz' % (spacename, nstates, Nparm, iq), DeltaF_ij=DeltaF_ij, dDeltaF_ij=dDeltaF_ij, DeltaU_ij=DeltaU_ij, dDeltaU_ij=dDeltaU_ij, DeltaS_ij=DeltaS_ij, dDeltaS_ij=dDeltaS_ij) #Save file
            else:
                DeltaF_file = numpy.load('hs_%s/ns%iNp%iQ%i.npz' % (spacename, nstates, Nparm, iq))
                DeltaF_ij = DeltaF_file['DeltaF_ij']
                dDeltaF_ij = DeltaF_file['dDeltaF_ij']
                DeltaU_ij = DeltaF_file['DeltaU_ij']
                dDeltaU_ij = DeltaF_file['dDeltaU_ij']
                DeltaS_ij = DeltaF_file['DeltaS_ij']
                dDeltaS_ij = DeltaF_file['dDeltaS_ij']
            #printFreeEnergy(DeltaF_ij, dDeltaF_ij)
            if not timekln:
                for iepsi in xrange(Nparm):
                    #if includeRef:
                    #    DelF[iq, iepsi,:] = DeltaF_ij[nstates,nstates+offset:]
                    #    dDelF[iq, iepsi,:] = dDeltaF_ij[nstates,nstates+offset:]
                    #Unwrap the data
                    DelF[iq, iepsi,:] = DeltaF_ij[0, 1 + iepsi*Nparm:1 + (iepsi+1)*Nparm]
                    dDelF[iq, iepsi,:] = dDeltaF_ij[0, 1 + iepsi*Nparm:1+ (iepsi+1)*Nparm]
                    DelU[iq, iepsi,:] = DeltaU_ij[0, 1 + iepsi*Nparm:1 + (iepsi+1)*Nparm]
                    dDelU[iq, iepsi,:] = dDeltaU_ij[0, 1 + iepsi*Nparm:1 + (iepsi+1)*Nparm]
                    DelS[iq, iepsi,:] = DeltaS_ij[0, 1 + iepsi*Nparm:1 + (iepsi+1)*Nparm]
                    dDelS[iq, iepsi,:] = dDeltaS_ij[0, 1 + iepsi*Nparm:1 + (iepsi+1)*Nparm]
            laptime = time.clock()
            # Show timing statistics. copied from Repex.py, copywrite John Chodera
            final_time = time.time()
            elapsed_time = final_time - initial_time
            estimated_time_remaining = (final_time - run_start_time) / (iteration) * (number_of_iterations - iteration)
            estimated_total_time = (final_time - run_start_time) / (iteration) * (number_of_iterations)
            estimated_finish_time = final_time + estimated_time_remaining
            print "Iteration took %.3f s." % elapsed_time
            print "Estimated completion in %s, at %s (consuming total wall clock time %s)." % (str(datetime.timedelta(seconds=estimated_time_remaining)), time.ctime(estimated_finish_time), str(datetime.timedelta(seconds=estimated_total_time)))
    else:
        #if os.path.isfile('es_%s/N%iRef%iOff%iEpsi%i.npz' % (spacename, Nparm, Ref_state, offset, Nparm-1)) and savedata: #Pull data from 
        if os.path.isfile('hs_%s/ns%iNp%iQ%i.npz' % (spacename, nstates, Nparm, Nparm-1)) and savedata: #Pull data from 
            DelF = numpy.zeros([Nparm, Nparm, Nparm])
            dDelF = numpy.zeros([Nparm, Nparm, Nparm])
            DelU = numpy.zeros([Nparm, Nparm, Nparm])
            dDelU = numpy.zeros([Nparm, Nparm, Nparm])
            DelS = numpy.zeros([Nparm, Nparm, Nparm])
            dDelS = numpy.zeros([Nparm, Nparm, Nparm])
            for iq in xrange(Nparm):
                sys.stdout.flush()
                sys.stdout.write('\rSave data detected, loading file %d/%d...' % (iq,Nparm-1))
                DeltaF_file = numpy.load('hs_%s/ns%iNp%iQ%i.npz' % (spacename, nstates, Nparm, iq))
                DeltaF_ij = DeltaF_file['DeltaF_ij']
                dDeltaF_ij = DeltaF_file['dDeltaF_ij']
                DeltaU_ij = DeltaF_file['DeltaU_ij']
                dDeltaU_ij = DeltaF_file['dDeltaU_ij']
                DeltaS_ij = DeltaF_file['DeltaS_ij']
                dDeltaS_ij = DeltaF_file['dDeltaS_ij']
                for iepsi in xrange(Nparm):
                    #if includeRef:
                    #    DelF[iq, iepsi,:] = DeltaF_ij[nstates,nstates+1:]
                    #    dDelF[iq, iepsi,:] = dDeltaF_ij[nstates,nstates+1:]
                    #Unwrap the data
                    DelF[iq, iepsi,:] = DeltaF_ij[0, 1 + iepsi*Nparm:1 + (iepsi+1)*Nparm]
                    dDelF[iq, iepsi,:] = dDeltaF_ij[0, 1 + iepsi*Nparm:1 + (iepsi+1)*Nparm]
                    DelU[iq, iepsi,:] = DeltaU_ij[0, 1 + iepsi*Nparm:1 + (iepsi+1)*Nparm]
                    dDelU[iq, iepsi,:] = dDeltaU_ij[0, 1 + iepsi*Nparm:1 + (iepsi+1)*Nparm]
                    DelS[iq, iepsi,:] = DeltaS_ij[0, 1 + iepsi*Nparm:1 + (iepsi+1)*Nparm]
                    dDelS[iq, iepsi,:] = dDeltaS_ij[0, 1 + iepsi*Nparm:1 + (iepsi+1)*Nparm]
            sys.stdout.write('\n')
    ###############################################
    ######### END FREE ENERGY CALCULATIONS ########
    ###############################################
    #Set up scaling
    C12_max = 3
    C12_min = 3E-5
    C6_max = 3
    C6_min = 1E-3
    plotC12_6 = True
    plotEpsiSig = True
    if plotC12_6 or plotEpsiSig:
        C12map = lambda x: ((C12_max - C12_min)*x + C12_min)*kjpermolTokcal
        C6map = lambda x: ((C6_max - C6_min)*x + C6_min)*kjpermolTokcal
        ylabel = r'$\epsilon$ in kcal/mol'
        if sig_factor != 3:
            xlabel = r'$\sigma$ in nm'
        else:
            xlabel = r'$\sigma^{%s}$ in nm$^{%s}$' % (sig_factor, sig_factor)
    else:
        C12map = lambda x: x
        C6map = lambda x: x
        xlabel = r'$\lambda$ of $C12_i$$'
        ylabel = r'$\lambda$ of $C6_i$$'
        
    DelF *= kjpermolTokcal/kjpermolTokT
    dDelF *= kjpermolTokcal/kjpermolTokT
    DelU *= kjpermolTokcal/kjpermolTokT
    dDelU *= kjpermolTokcal/kjpermolTokT
    DelS *= kjpermolTokcal/kjpermolTokT  / (T/units.kelvin)
    dDelS *= kjpermolTokcal/kjpermolTokT / (T/units.kelvin)
    #Relative error
    #reldDelF = numpy.abs(dDelF/DelF)
    #pdb.set_trace()
 

    ################################################
    ################# PLOTTING #####################
    ################################################
    #Plot the sigma and epsilon free energies
    relativeErr = False
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
    #Entropy
    h,(Hplot,dHplot) = plt.subplots(2,1,sharex=True)
    gH,rdHplot = plt.subplots(1,1)
    rdHFig = gH
    Hplotlist=[Hplot,dHplot]
    s,(Splot,dSplot) = plt.subplots(2,1,sharex=True)
    gS,rdHplot = plt.subplots(1,1)
    rdSFig = gH
    Splotlist=[Splot,dSplot]
    import matplotlib.animation as ani
    cvmax = DelF.max()*1.01
    cvmin = DelF.min()*1.01
    cvmaxH = DelU.max()*1.01
    cvminH = DelU.min()*1.01
    cvmaxS = DelS.max()*1.01
    cvminS = DelS.min()*1.01
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
    #load for H and S
    try: # H
        errorlimitsH = numpy.load('n24H_error_lims.npy')
        cdvminH = errorlimitsH[0]
        cdvmaxH = errorlimitsH[1]
    except:
        #cdvminH = dDelU.min()*1.01
        #cdvmaxH = dDelU.max()*1.01
        cdvminH = numpy.nanmin(dDelU)*0.99
        cdvmaxH = numpy.nanmax(dDelU)*1.01
        cdvmaxH = numpy.nanmean(dDelU) + 0.1*numpy.nanstd(dDelU)
        #Set to same scale as dDelF at n21
        if nstates == 21:        
            cdvmaxH = 53.405545268981086 * 1.01
        else:
            cdvmaxH = numpy.nanmax(dDelU)*1.01
     
        if nstates == 24:
            numpy.save('n24H_error_lims.npy', numpy.array([dDelU.min()*1.01, dDelU.max()*1.01]))
    try: #S
        errorlimitsS = numpy.load('n24S_error_lims.npy')
        cdvminS = errorlimitsS[0]
        cdvmaxS = errorlimitsS[1]
    except:
        #cdvminS = dDelS.min()*1.01
        #cdvmaxS = dDelS.max()*1.01
        cdvminS = numpy.nanmin(dDelS)*0.99
        cdvmaxS = numpy.nanmax(dDelS)*1.01
        cdvmaxS = numpy.nanmean(dDelS) + 0.1*numpy.nanstd(dDelS)
        if nstates == 21:        
            cdvmaxS = 53.405545268981086/298 * 1.01
        else:
            cdvmaxS = numpy.nanmax(dDelS)*1.01
        if nstates == 24:
            numpy.save('n24S_error_lims.npy', numpy.array([dDelS.min()*1.01, dDelS.max()*1.01]))
    if relativeErr:
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

    q_title_template = r"$q=%.2f$"
    #Plot H#
    Hqtitle = h.text(0.85, 0.95, '', fontsize=20)
    h.text(0.98, .71, r'$\Delta H$', rotation=-90, horizontalalignment='center', verticalalignment='center', fontsize=21)
    h.text(0.98, .27, r'$\delta\left(\Delta H\right)$', rotation=-90, horizontalalignment='center', verticalalignment='center', fontsize=21)
    imgHplot = Hplot.pcolormesh(sig_range**sig_factor,epsi_range,DelU[(Nparm-1)/2,:,:])
    #Set the colorbar
    divHplot = mal(Hplot)
    caxHplot = divHplot.append_axes('right', size='5%', pad=0.05)
    cHplot = h.colorbar(imgHplot, cax=caxHplot)
    #set the minmax colorscales
    ####### Error plot #######
    imgdHplot = dHplot.pcolormesh(sig_range**sig_factor,epsi_range,dDelU[(Nparm-1)/2,:,:])
    divdHplot = mal(dHplot)
    caxdHplot = divdHplot.append_axes('right', size='5%', pad=0.05)
    #Set the minmax colorscales
    imgdHplot.set_clim(vmin=cdvminH, vmax=cdvmaxH)
    cdHplot = h.colorbar(imgdHplot, cax=caxdHplot)
    Hsup_title_template = r'$\Delta H$ (top) and $\delta\Delta H$(bottom) with $q=%.2f$ for LJ Spheres' + '\n in units of kcal/mol'
    htitle = h.suptitle('')
    #Set up the empty plots
    #Hscatters = []
    #dHscatters = []
    Hline, = Hplot.plot([], [], linewidth=2, color='k')
    dHline, = dHplot.plot([], [], linewidth=2, color='w')
    #H_scatter_noref = Hplot.scatter([], [], s=60, c='k', marker='x')
    #dH_scatter_noref = dHplot.scatter([], [], s=60, c='w', marker='x')
    #H_scatter_ref = Hplot.scatter([], [], s=70, c='k', marker='D')
    #dH_scatter_ref = dHplot.scatter([], [], s=70, c='w', marker='D')
    H_scatter_noref, = Hplot.plot([], [], linestyle='', markersize=5, color='k', marker='x', markeredgewidth=2)
    dH_scatter_noref, = dHplot.plot([], [], linestyle='', markersize=5, color='w', marker='x', markeredgewidth=2)
    H_scatter_ref, = Hplot.plot([], [], linestyle='', markersize=6, color='k', marker='D', markeredgewidth=2)
    dH_scatter_ref, = dHplot.plot([], [], linestyle='', markersize=6, color='w', marker='D', markeredgewidth=2, markeredgecolor='w')
    #Create the scatter sampled data
    noref = numpy.ma.array(range(nstates), mask=False)
    noref.mask[Ref_state]=True
    noref = noref.compressed()
    scatter_epsis = epsi_samp_space[noref]
    scatter_sig = sig_samp_space[noref]
    for ax in Hplotlist:
        ax.set_yscale(spacename)
        ax.set_xscale(spacename)
        ax.set_ylim([epsiPlotStart,epsiPlotEnd])
        ax.set_xlim([sigPlotStart,sigPlotEnd])
        ax.patch.set_color('grey')
    h.subplots_adjust(hspace=0.02)
    h.text(0.05, .5, ylabel, rotation='vertical', horizontalalignment='center', verticalalignment='center', fontsize=20)
    dHplot.set_xlabel(xlabel, fontsize=20)
    #Animate the figures
    realcMaxH = -10**20 #Some really small number
    realcMinH = 10**20 #some really large number
    realdcMaxH = -10**20 #Some really small number
    realdcMinH = 10**20 #some really large number
    q_ref = q_samp_space[Ref_state]
    sig_ref = sig_samp_space[Ref_state]
    epsi_ref = epsi_samp_space[Ref_state]
    def cleanupH():
        imgHplot.set_array([])
        imgdHplot.set_array([])
        #imgrdHplot.set_array([])
        htitle.set_text('')
        H_scatter_ref.set_data([], [])
        dH_scatter_ref.set_data([], [])
        H_scatter_noref.set_data([], [])
        dH_scatter_noref.set_data([], [])
        Hline.set_data([], [])
        dHline.set_data([], [])
        #for iscatter,discatter in zip(Fscatters,dFscatters):
        #    iscatter.set_data([],[])
        #    discatter.set_data([],[])
        Hqtitle.set_text('')
        if relativeErr:
            return imgFplot, imgdFplot, imgrdFplot, ftitle, F_scatter_noref, dF_scatter_noref, F_scatter_ref, dF_scatter_ref, Fline, dFline
        else:
            return imgHplot, imgdHplot, htitle, H_scatter_noref, dH_scatter_noref, H_scatter_ref, dH_scatter_ref, Hline, dHline, Hqtitle
    def moveqH(qndx):
        q = q_range[qndx]
        #I have to create a secondary pcolormesh set since pcolormesh truncates the array size to make the render faster (I dont think pcolor would but would be slower)
        #If you don't do this, then it creates a weird set of lines which make no sense
        scrapHig, (scrapH, scrapdH, scraprdH) = plt.subplots(3,1)
        scrapHplot = scrapH.pcolormesh(sig_range**sig_factor,epsi_range,DelU[qndx,:,:])
        #Lock down the color choice for the error plot
        scrapdHplot = scrapdH.pcolormesh(sig_range**sig_factor,epsi_range,dDelU[qndx,:,:], vmax=cdvmaxH, vmin=cdvminH)
        #Lock down the color choice for the relative error plot
        #scraprdHplot = scraprdH.pcolormesh(sig_range**sig_factor,epsi_range,reldDelU[qndx,:,:],vmax=crdvmaxH, vmin=crdvminH)
        #Reassign the plots, if you did not use an already generated array, you would need to .ravel() on the array you feed to set_array()
        imgHplot.set_array(scrapHplot.get_array())
        imgdHplot.set_array(scrapdHplot.get_array())
        #imgrdHplot.set_array(scraprdHplot.get_array())
        #htitle.set_text(Hsup_title_template % q)
        Hqtitle.set_text(q_title_template % q)
        Hmax = DelU[qndx,:,:].max()
        Hmin = DelU[qndx,:,:].min()
        dHmax = dDelU[qndx,:,:].max()
        dHmin = dDelU[qndx,:,:].min()
        imgHplot.set_clim(vmin=Hmin, vmax=Hmax)
        #imgdFplot.set_clim(vmin=dDmin, vmax=dDmax)
        #Set up the scatters
        #Set the Q scatters correctly
        qsampled = numpy.where(q_samp_space == q) #Get all sampled states from the current q value
        epsi_qsamp = epsi_samp_space[qsampled]
        sig_qsamp = sig_samp_space[qsampled]
        H_scatter_noref.set_data(sig_qsamp**sig_factor, epsi_qsamp)
        dH_scatter_noref.set_data(sig_qsamp**sig_factor, epsi_qsamp)
        if numpy.any(Ref_state == qsampled[0]):
            H_scatter_ref.set_data(sig_ref**sig_factor, epsi_ref)
            dH_scatter_ref.set_data(sig_ref**sig_factor, epsi_ref)
        else:
            #H_scatter_ref.set_data([], [])
            #dH_scatter_ref.set_data([], [])
            H_scatter_ref.set_data(sig_ref**sig_factor, epsi_ref)
            dH_scatter_ref.set_data(sig_ref**sig_factor, epsi_ref)
        if q == 0:
            Hline.set_data(sig_range**sig_factor, epsi_plot_range)
            dHline.set_data(sig_range**sig_factor, epsi_plot_range)
        else:
            Hline.set_data([], [])
            dHline.set_data([], [])
        Hline.set_data([], [])
        dHline.set_data([], [])
        H_scatter_ref.set_data([], [])
        dH_scatter_ref.set_data([], [])
        H_scatter_noref.set_data([], [])
        dH_scatter_noref.set_data([], [])
        #Cleanup scrap figure to avoid memory buildup
        plt.close(scrapHig)
        if relativeErr:
            return imgFplot, imgdFplot, imgrdFplot, ftitle, F_scatter_noref, dF_scatter_noref, F_scatter_ref, dF_scatter_ref, Fline, dFline
        else:
            #return imgHplot, imgdHplot, htitle, H_scatter_noref, dH_scatter_noref, H_scatter_ref, dH_scatter_ref, Hline, dHline
            return imgHplot, imgdHplot, H_scatter_noref, dH_scatter_noref, H_scatter_ref, dH_scatter_ref, Hline, dHline, Hqtitle
    aniH = ani.FuncAnimation(h, moveqH, range(Nparm), interval=150, blit=False, init_func=cleanupH)
    if relativeErr:
        filename='Animated_charging_rel{myint:{width}}.mp4'.format(myint=nstates, width=len(str(nstates)))
    else:
        filename='Animated_charging_H{myint:{width}}.mp4'.format(myint=nstates, width=len(str(nstates)))
    #pdb.set_trace()
    aniH.save(filename, dpi=400)
    #save a single frame
    qframes=[11, 24]
    for qframe in qframes:
        moveqH(qframe)
        h.savefig('SingleFrameH_n%d_%d.png' % (nstates, qframe), bbox_inches='tight', dpi=600)
    #pdb.set_trace()


    """
    Repeat process for S
    """
    #Plot s
    Sqtitle = s.text(0.85, 0.95, '', fontsize=20)
    s.text(0.98, .71, r'$\Delta S$', rotation=-90, horizontalalignment='center', verticalalignment='center', fontsize=21)
    s.text(0.98, .27, r'$\delta\left(\Delta S\right)$', rotation=-90, horizontalalignment='center', verticalalignment='center', fontsize=21)
    imgSplot = Splot.pcolormesh(sig_range**sig_factor,epsi_range,DelS[(Nparm-1)/2,:,:])
    #Set the colorbar
    divSplot = mal(Splot)
    caxSplot = divSplot.append_axes('right', size='5%', pad=0.05)
    cSplot = s.colorbar(imgSplot, cax=caxSplot)
    #set the minmax colorscales
    ####### Error plot #######
    imgdSplot = dSplot.pcolormesh(sig_range**sig_factor,epsi_range,dDelS[(Nparm-1)/2,:,:])
    divdSplot = mal(dSplot)
    caxdSplot = divdSplot.append_axes('right', size='5%', pad=0.05)
    #Set the minmax colorscales
    imgdSplot.set_clim(vmin=cdvminS, vmax=cdvmaxS)
    cdSplot = s.colorbar(imgdSplot, cax=caxdSplot)
    Ssup_title_template = r'$\Delta S$ (top) and $\delta\Delta S$(bottom) with $q=%.2f$ for LJ Spheres' + '\n in units of kcal/(K$\cdot$mol)'
    stitle = s.suptitle('')
    #Set up the empty plots
    #Sscatters = []
    #dSscatters = []
    Sline, = Splot.plot([], [], linewidth=2, color='k')
    dSline, = dSplot.plot([], [], linewidth=2, color='w')
    #S_scatter_noref = Splot.scatter([], [], s=60, c='k', marker='x')
    #dS_scatter_noref = dSplot.scatter([], [], s=60, c='w', marker='x')
    #S_scatter_ref = Splot.scatter([], [], s=70, c='k', marker='D')
    #dS_scatter_ref = dSplot.scatter([], [], s=70, c='w', marker='D')
    S_scatter_noref, = Splot.plot([], [], linestyle='', markersize=5, color='k', marker='x', markeredgewidth=2)
    dS_scatter_noref, = dSplot.plot([], [], linestyle='', markersize=5, color='w', marker='x', markeredgewidth=2)
    S_scatter_ref, = Splot.plot([], [], linestyle='', markersize=6, color='k', marker='D', markeredgewidth=2)
    dS_scatter_ref, = dSplot.plot([], [], linestyle='', markersize=6, color='w', marker='D', markeredgewidth=2, markeredgecolor='w')
    #Create the scatter sampled data
    noref = numpy.ma.array(range(nstates), mask=False)
    noref.mask[Ref_state]=True
    noref = noref.compressed()
    scatter_epsis = epsi_samp_space[noref]
    scatter_sig = sig_samp_space[noref]
    for ax in Splotlist:
        ax.set_yscale(spacename)
        ax.set_xscale(spacename)
        ax.set_ylim([epsiPlotStart,epsiPlotEnd])
        ax.set_xlim([sigPlotStart,sigPlotEnd])
        ax.patch.set_color('grey')
    s.subplots_adjust(hspace=0.02)
    s.text(0.05, .5, ylabel, rotation='vertical', horizontalalignment='center', verticalalignment='center', fontsize=20)
    dSplot.set_xlabel(xlabel, fontsize=20)
    #Animate the figures
    realcMaxS = -10**20 #Some really small number
    realcMinS = 10**20 #some really large number
    realdcMaxS = -10**20 #Some really small number
    realdcMinS = 10**20 #some really large number
    q_ref = q_samp_space[Ref_state]
    sig_ref = sig_samp_space[Ref_state]
    epsi_ref = epsi_samp_space[Ref_state]
    def cleanupS():
        imgSplot.set_array([])
        imgdSplot.set_array([])
        #imgrdSplot.set_array([])
        stitle.set_text('')
        S_scatter_ref.set_data([], [])
        dS_scatter_ref.set_data([], [])
        S_scatter_noref.set_data([], [])
        dS_scatter_noref.set_data([], [])
        Sline.set_data([], [])
        dSline.set_data([], [])
        Sqtitle.set_text('')
        #for iscatter,discatter in zip(Fscatters,dFscatters):
        #    iscatter.set_data([],[])
        #    discatter.set_data([],[])
        if relativeErr:
            return imgFplot, imgdFplot, imgrdFplot, ftitle, F_scatter_noref, dF_scatter_noref, F_scatter_ref, dF_scatter_ref, Fline, dFline
        else:
            return imgSplot, imgdSplot, htitle, S_scatter_noref, dS_scatter_noref, S_scatter_ref, dS_scatter_ref, Sline, dSline
    def moveqS(qndx):
        q = q_range[qndx]
        #I have to create a secondary pcolormesh set since pcolormesh truncates the array size to make the render faster (I dont think pcolor would but would be slower)
        #If you don't do this, then it creates a weird set of lines which make no sense
        scrapSig, (scrapS, scrapdS, scraprdS) = plt.subplots(3,1)
        scrapSplot = scrapS.pcolormesh(sig_range**sig_factor,epsi_range,DelS[qndx,:,:])
        #Lock down the color choice for the error plot
        scrapdSplot = scrapdS.pcolormesh(sig_range**sig_factor,epsi_range,dDelS[qndx,:,:], vmax=cdvmaxS, vmin=cdvminS)
        #Lock down the color choice for the relative error plot
        #scraprdSplot = scraprdS.pcolormesh(sig_range**sig_factor,epsi_range,reldDelS[qndx,:,:],vmax=crdvmaxS, vmin=crdvminS)
        #Reassign the plots, if you did not use an already generated array, you would need to .ravel() on the array you feed to set_array()
        imgSplot.set_array(scrapSplot.get_array())
        imgdSplot.set_array(scrapdSplot.get_array())
        #imgrdSplot.set_array(scraprdSplot.get_array())
        #stitle.set_text(Ssup_title_template % q)
        Sqtitle.set_text(q_title_template % q)
        Smax = DelS[qndx,:,:].max()
        Smin = DelS[qndx,:,:].min()
        dSmax = dDelS[qndx,:,:].max()
        dSmin = dDelS[qndx,:,:].min()
        imgSplot.set_clim(vmin=Smin, vmax=Smax)
        #imgdFplot.set_clim(vmin=dDmin, vmax=dDmax)
        #Set up the scatters
        #Set the Q scatters correctly
        qsampled = numpy.where(q_samp_space == q) #Get all sampled states from the current q value
        epsi_qsamp = epsi_samp_space[qsampled]
        sig_qsamp = sig_samp_space[qsampled]
        S_scatter_noref.set_data(sig_qsamp**sig_factor, epsi_qsamp)
        dS_scatter_noref.set_data(sig_qsamp**sig_factor, epsi_qsamp)
        if numpy.any(Ref_state == qsampled[0]):
            S_scatter_ref.set_data(sig_ref**sig_factor, epsi_ref)
            dS_scatter_ref.set_data(sig_ref**sig_factor, epsi_ref)
        else:
            #S_scatter_ref.set_data([], [])
            #dS_scatter_ref.set_data([], [])
            S_scatter_ref.set_data(sig_ref**sig_factor, epsi_ref)
            dS_scatter_ref.set_data(sig_ref**sig_factor, epsi_ref)
        if q == 0:
            Sline.set_data(sig_range**sig_factor, epsi_plot_range)
            dSline.set_data(sig_range**sig_factor, epsi_plot_range)
        else:
            Sline.set_data([], [])
            dSline.set_data([], [])
        Sline.set_data([], [])
        dSline.set_data([], [])
        S_scatter_ref.set_data([], [])
        dS_scatter_ref.set_data([], [])
        S_scatter_noref.set_data([], [])
        dS_scatter_noref.set_data([], [])
        #Cleanup scrap figure to avoid memory buildup
        plt.close(scrapSig)
        if relativeErr:
            return imgFplot, imgdFplot, imgrdFplot, ftitle, F_scatter_noref, dF_scatter_noref, F_scatter_ref, dF_scatter_ref, Fline, dFline
        else:
            #return imgSplot, imgdSplot, stitle, S_scatter_noref, dS_scatter_noref, S_scatter_ref, dS_scatter_ref, Sline, dSline
            return imgSplot, imgdSplot, Sqtitle, S_scatter_noref, dS_scatter_noref, S_scatter_ref, dS_scatter_ref, Sline, dSline
    aniS = ani.FuncAnimation(s, moveqS, range(Nparm), interval=150, blit=False, init_func=cleanupS)
    if relativeErr:
        filename='Animated_charging_rel{myint:{width}}.mp4'.format(myint=nstates, width=len(str(nstates)))
    else:
        filename='Animated_charging_S{myint:{width}}.mp4'.format(myint=nstates, width=len(str(nstates)))
    #pdb.set_trace()
    aniS.save(filename, dpi=400)
    #save a single frame
    #qframe=40
    #moveqS(qframe)
    #s.savefig('DelS_Nstate_%i_Qndx_%i.eps' % (nstates, qframe), bbox_inches='tight', dpi=400)
    qframes=[11, 24]
    for qframe in qframes:
        moveqS(qframe)
        s.savefig('SingleFrameS_n%d_%d.png' % (nstates, qframe), bbox_inches='tight', dpi=600)
    #pdb.set_trace()

####################################################################################
####################################################################################
####################################################################################

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--nstates", dest="nstates", default=None, help="Set the number of states", metavar="NSTATES")
    #nstate options: 24, 32, 40, 49
    (options, args) = parser.parse_args()
    x = numpy.load('qes.npy')
    qs = x[:,0]
    es = x[:,1]
    ss = x[:,2]
    if options.nstates is None:
        nstates = len(qs)
    else:
        nstates = int(options.nstates)
    pdb.set_trace()
    execute(nstates, qs, es, ss)
