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
#import dbscan
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph
import MDAnalysis as mda


#################### OPTIONS ####################
relativeErr = False #Create the relative error plot
savedata = True #Save/load dhdl data
load_ukln = True #Save/load u_kln from file
timekln = False #Time energy evaluation, skips actual free energy evaluation
graphsfromfile=True #Generate graphs from the .npz arrays
load_rdf = False #Try to load RDF data or generate it yourself

Ref_state = 1 #Reference state of sampling to pull from

################## END OPTIONS ##################




kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
T = 298 * units.kelvin
kT = kB*T
kjpermolTokT = units.kilojoules_per_mole / kT
kjpermolTokcal = 1/4.184

spacing=linspace

#Default binning
defbin = 800/5

#RDF range in min and max angstroms
distmin = 0 
distmax = 12
defaultdist = (distmin, distmax)

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


def buildrdf(stateindex, nbins=defbin, histrange=defaultdist, useO=False, fulltrr=False):
    #Create the universe
    base_path = os.path.join('/mnt', 'argon', 'data', 'ln8dc', 'ljsphere_es', 'lj%s' % stateindex, 'prod')
    topol_path = os.path.join(base_path, 'prod%s.gro' % stateindex)
    trrfile = 'prod%s.trr'
    if not fulltrr: 
        trrfile = 'sub' + trrfile
    traj_path = os.path.join(base_path, trrfile % stateindex)
    universe = mda.Universe(topol_path, traj_path)
    #Set the two systems
    #The coordinate call to each of these updates with the updated timestep (ts) of the universe.trajectory
    #So subsequent X.coordinates() will return the current timestep coordinates
    if useO:
        solvent = universe.selectAtoms('resname SOL and name OW') #Get the oxygen distance
    else:
        solvent = universe.selectAtoms('resname SOL') #Get the whole water molecule
    particle = universe.selectAtoms('name LJS')
    #Initilize array
    frames = universe.trajectory.numframes
    rdfs = numpy.zeros([nbins, frames])
    #Go through each frame
    iframe = 0
    for ts in universe.trajectory:
        #Get Box dimensions
        box=ts.dimensions[:3]
        #Get the coordinates for each group 
        ljscoords = particle.coordinates()
        solvcoords = solvent.coordinates()
        #Get the distances
        dist = mda.core.distances.distance_array(ljscoords,solvcoords,box)
        rdf, edges = numpy.histogram(dist, bins=nbins, range=histrange)
        rdf = rdf.astype(numpy.float64)
        #normalize this frame
        boxvol = ts.volume
        radii = 0.5 * (edges[1:] + edges[:-1])
        numvol = solvent.numberOfAtoms() / boxvol
        norm = (4.0/3.0) * numpy.pi * (numpy.power(edges[1:],3) - numpy.power(edges[:-1],3))
        rdf /= norm*numvol 
        rdfs[:,iframe] = rdf
        iframe += 1 
    return rdfs

def execute(nstates, nbins=defbin, histrange=defaultdist, useO=False):
    #Generate empty histogram array: indexed: bin, k, frame
    totalhist = numpy.zeros([nbins, nstates, 0], dtype=numpy.float64)
    if useO:
        pathtail = 'b%s'%nbins
        pathstr = 'lj%s/prod/rdfOhist%s.npz'
        pathstr = 'lj%s/prod/rdfOhist%s' + pathtail + '.npz'
    else:
        pathstr = 'lj%s/prod/rdfhist%s.npz'
    #Cycle through each state
    for k in xrange(nstates):
        #Try to load in the RDF file
        if load_rdf and os.path.isfile(pathstr%(k,k)):
            rdfhist = numpy.load(pathstr%(k,k))['rdf']
        else:
            print "Generating RDF for state %d" % k
            rdfhist = buildrdf(k, nbins=nbins, histrange=histrange, useO=useO)
            if load_rdf:
                savez(pathstr%(k,k), rdf=rdfhist)
    #f,a = plt.subplots(1,1)
    #xaxis = linspace(histrange[0], histrange[1], nbins)/10.0 #Convert to NM
    #a.plot(xaxis, rdfhist.sum(axis=1)/float(rdfhist.shape[1]))
    #a.set_ylim([0,2])
    #plt.show()
    #pdb.set_trace()

####################################################################################
####################################################################################
####################################################################################

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--nstates", dest="nstates", default=None, help="Set the number of states", metavar="NSTATES")
    parser.add_option("--nbins", dest="nbins",  type='int', default=defbin, help="Set the number of histogram bins: Default %d" % defbin, metavar="NBINS")
    parser.add_option("--minA", dest="minA", default=distmin, help="Set the histogram minimum. Default: %f Ang" % distmin, metavar="MINA")
    parser.add_option("--maxA", dest="maxA", default=distmax, help="Set the histogram maximum. Default: %f Ang" % distmax, metavar="MAXA")
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
    #If this script is run with manaual execution, break here since its probably being run as debugging anyways.
    pdb.set_trace()
    execute(nstates, nbins=options.nbins, histrange=(options.minA, options.maxA), useO=options.oxygen)
