import numpy
import matplotlib.pyplot as plt
from scipy import linspace
import gen_rdf_data as genrdf
import pdb
import os.path

ions = ['Li+', 'Cl-']
ions = ['Li+',      'Na+',       'K+',        'Rb+',       'Cs+',       'F-',        'Cl-',       'Br-',       'I-']
nions = len(ions)
nstates = 203
filestring = 'pertrdfs/pertrdfOn%s%s.npz'
basebins = genrdf.defbin

savefig = False

Erdfs = {}
dErdfs = {}
figs = {}
plots = {}

plotx = linspace(genrdf.distmin, genrdf.distmax, basebins)/10.0 #put in nm

for l in xrange(nions):
    ion = ions[l]
    rdfdata = numpy.load(filestring % (nstates, ion))
    Erdfs[ion] = rdfdata['Erdfs']
    dErdfs[ion] = rdfdata['dErdfs']
    figs[ion], plots[ion] = plt.subplots(1,1)
    p = plots[ion]
    f = figs[ion]
    p.plot(plotx, Erdfs[ion], '-k', linewidth = 2)
    p.plot(plotx, Erdfs[ion]+dErdfs[ion], '--k', linewidth = 1)
    p.plot(plotx, Erdfs[ion]-dErdfs[ion], '--k', linewidth = 1)
    p.set_ylim([0,plots[ion].get_ylim()[1]])
    p.set_xlabel(r'$r$ in nm', fontsize=15)
    p.set_ylabel(r'$g(r)_{\mathrm{Ion-O}}$', fontsize=17)
    if savefig:
        f.patch.set_alpha(0)
        p.patch.set_alpha(1)
        p.patch.set_color('w')
        f.savefig('rdf%s.eps'%ion, bbox_inches='tight')
    #Print the rdf=0 location
    #Make a blank array of 0 for comparison with the isclose command
    blank0 = numpy.zeros(Erdfs[ion].shape)
    #x0max = plotx[numpy.where(Erdfs[ion]==0)].max()
    x0max = plotx[numpy.isclose(Erdfs[ion], blank0)].max()
    print "Max location where %s RDF is 0: r=%f" % (ion, x0max)
if not savefig:
    #plt.show()
    pdb.set_trace()
