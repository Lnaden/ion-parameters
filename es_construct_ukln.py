from basisanalyze import *
import numpy
from numpy import ma
from scipy.integrate import simps
from scipy import linspace
from scipy import logspace
import scipy.optimize
import matplotlib.pyplot as plt
import os.path
import pdb
import simtk.unit as units
from pymbar import MBAR
from numpy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
from matplotlib import ticker
import time
import datetime
import sys
from mpl_toolkits.mplot3d import axes3d

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
T = 298 * units.kelvin
kT = kB*T
kjpermolTokT = units.kilojoules_per_mole / kT
kjpermolTokcal = 1/4.184
graphsfromfile=True #Generate graphs from the .npz arrays
savedata = True #Save dhdl data
masked=False

#logspace or linspace
narg = len(sys.argv)
if narg > 1:
    if sys.argv[1] == 'log':
        spacing=logspace
    elif sys.argv[1] == 'lin':
        spacing=linspace
    else:
        spacing=linspace
else:
    spacing=linspace

#Main output controling vars:
nstates = 12
Nparm = 151 #101 or 151
plotReal = True
sig_factor=1
annotatefig = True
savefigs = True
if Nparm == 151:
    alle = True
else:
   alle = False

sig_min = 0.1 #nm
sig_max = 1.2
sig3_samp_space = linspace(sig_min**3, sig_max**3, nstates)
sig_samp_space = sig3_samp_space**(1.0/3)
sig_min = 0.25 #updated for 
sig_samp_space[0] = sig_min
sig3_samp_space[0] = sig_samp_space[0]**3
epsi_min = 0.1 #kJ/mol
epsi_max = 1.2
epsi_samp_space = linspace(epsi_min, epsi_max, 11)
lamto_epsi = lambda lam: (epsi_max - epsi_min)*lam + epsi_min
lamto_sig3 = lambda lam: (sig_max**3 - sig_min**3)*lam + sig
lamto_sig = lambda lam: lamto_sig3(lam)**(1.0/3)


#epsi_samp_space = numpy.array([0.100, 6.960, 2.667, 1.596, 1.128, 0.870, 0.706, 0.594, 0.513, 0.451, 0.40188])
#sig_samp_space = numpy.array([0.25000, 0.41677, 0.58856, 0.72049, 0.83175, 0.92978, 1.01843, 1.09995, 1.17584, 1.24712, 1.31453])
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#This is the true sigma sampling space because I made a small math error when initially spacing 
sig_samp_space = numpy.array([0.25, 0.57319535, 0.712053172, 0.811158734, 0.890612296, 0.957966253, 1.016984881, 1.069849165, 1.117949319, 1.162232374, 1.2, 0.3])
epsi_samp_space = numpy.append(epsi_samp_space, 0.8)
sig3_samp_space = sig_samp_space**3
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#Real molecule sampling
realname = ['UAm', 'NOP', 'C60', 'LJ6', 'null', 'LJ11']
nreal = len(realname)
realepsi = numpy.array([1.2301, 3.4941, 1.0372, 0.7600, 0, 0.8])
realsig  = numpy.array([0.3730, 0.6150, 0.9452, 1.0170, 0, 0.3])

#epsi_max = 0.40188
#sig_max = 1.31453

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
    if alle:
        epsiEndSpace   = 3.6 #!!! Manual set
    else:
        epsiEndSpace   = epsi_max
    spacename='linear'
    sigPlotStart = sigStartSpace**sig_factor
    sigPlotEnd   = sigEndSpace**sig_factor
    epsiPlotStart = epsiStartSpace
    epsiPlotEnd   = epsiEndSpace


################ SUBROUTINES ##################
def my_annotate(ax, s, xy_arr=[], fontsize=12, *args, **kwargs):
  ans = []
  an = ax.annotate(s, xy_arr[0], fontsize=fontsize, *args, **kwargs)
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


def cubic_mono_hermite_spline(x,y):
    #Create a sequence of cubic hermite splines
    #Code based on Michalski's Python variant
    n = len(x)
    #Compute Secants
    secants = (y[1:] - y[:-1])/ (x[1:] - x[:-1])
    #Compute initial tangents
    tangents = numpy.zeros(n)
    tangents[0] = secants[0]
    tangents[-1] = secants[-1]
    tangents[1:-1] = (secants[:-1]+secants[1:])/2
    #Solve case where delta = 0
    m_to_change = numpy.compress((secants == 0.0), range(n))
    for i in m_to_change:
        tangents[i] = 0.0
        tangents[i+1] = 0.0
    #Create alpha and beta
    alpha = tangents[:-1] / secants
    beta = tangents[1:] / secants
    distance = alpha**2 + beta**2
    tau = 3.0 / numpy.sqrt(distance)
    #Find where the alpha and beta cannot be transcribed within a guarenteed monotonic circle of radius 3
    over = (distance > 9.0)
    m_to_change = numpy.compress(over, range(n))
    #Find where there is non monotonics
    notmono = numpy.logical_or(alpha < 0, beta < 0)
    m_to_mono = numpy.compress(notmono, range(n))
    for j in m_to_mono:
       tangents[j] = 0
       tangents[j+1] = 0
    #Build the monotonics
    for i in m_to_change:
        #check to see if i is in m_to_mono and dont touch it if it is
        if i in m_to_mono:
            continue
        else:
            tangents[i] = tau[i] * alpha[i] * secants[i]
            tangents[i+1] = tau[i] * beta[i] * secants[i]
    return tangents

def hermite_spline(x_in,y_min,y_max,m_min,m_max):
    #Build a set of y values to pass back over the hermite spline
    x_min = x_in[0]
    x_max = x_in[-1]
    h = x_max - x_min
    t = (x_in - x_min)/h
    h00 = (1+2*t) * (1-t)**2
    h10 = t * (1-t)**2
    h01 = t**2 * (3-2*t)
    h11 = t**2 * (t-1)
    #dh/dx = dh/dt * dt/dx, dt/dx = 1/h
    dh00 = (6*t**2 - 6*t)/h
    dh10 = (3*t**2 - 4*t + 1)/h
    dh01 = (-6*t**2 + 6*t)/h
    dh11 = (3*t**2 - 2*t)/h
    y = y_min*h00 + h*m_min*h10 + y_max*h01 + h*m_max*h11
    dy = y_min*dh00 + h*m_min*dh10 + y_max*dh01 + h*m_max*dh11
    return y,dy
def buildHermite(x,y,n_between):
    #Find the tangents (will be needed)
    m = cubic_mono_hermite_spline(x,y)
    n = len(x)
    #Create the sequence of intermediate points to fill in the gaps
    x_filled = numpy.empty(0,numpy.float64)
    yr_filled = numpy.empty(0,numpy.float64)
    dyr_filled = numpy.empty(0,numpy.float64)
    for i in range(n-1):
        #Create the spacing
        x_to_hermite = scipy.linspace(x[i],x[i+1],n_between+2)
        (yr_hermite_out,dyr_herm_out) = hermite_spline(x_to_hermite,y[i],y[i+1],m[i],m[i+1])
        x_filled = numpy.append(x_filled[:-1], x_to_hermite)
        yr_filled = numpy.append(yr_filled[:-1], yr_hermite_out)
        dyr_filled = numpy.append(dyr_filled[:-1], dyr_herm_out)
    return x_filled,yr_filled,dyr_filled
#---------------------------------------------------------------------------------------------

def generalvar(g_in, lam_range, Ng, const_matricies, sourcesys, returnpoints=False, method='hermite', penaltytol=0, verbose=False, interm_n=2, penaltyK=500000, return_basis_only=False):
    #Split inputs
    gE = numpy.concatenate( ([0.0],g_in[0*Ng:1*Ng],[1.0]) )
    gR = numpy.concatenate( ([0.0],g_in[1*Ng:2*Ng],[1.0]) )
    gA = numpy.concatenate( ([0.0],g_in[2*Ng:3*Ng],[1.0]) )
    #Correct for small negative results
    for gX in [gE,gR,gA]:
        itszero = numpy.array([numpy.allclose(x,0.0) for x in gX])
        gX[itszero] = 0.0
        gX[numpy.where(gX < 0)] = 0
    print gE
    print gR
    print gA
    #If points are non-monotonic, apply a half-harmonic penalty
    penalty = 0
    for gX in [gE,gR,gA]:
        for i in xrange(len(gX)-2):#truncation
            j = i+1 #set gX index
            deltaleft = gX[j] - gX[j-1]
            deltaright = gX[j+1] - gX[j]
            if deltaleft - penaltytol< 0:
                penalty += penaltyK * (deltaleft-penaltytol)**2
            if deltaright -penaltytol < 0:
                penalty += penaltyK * (deltaright-penaltytol)**2
    #build the spline
    #spline = ius(lam_range,gE)
    #Derivatives
    (xHEout, yHEout, dHE) = buildHermite(lam_range,gE,interm_n) #the zero makes no extra points
    (xHRout, yHRout, dHR) = buildHermite(lam_range,gR,interm_n) #the zero makes no extra points
    (xHAout, yHAout, dHA) = buildHermite(lam_range,gA,interm_n) #the zero makes no extra points
    xHout=xHEout #They should be the same and therefore should not mater
    xHoutAll = {'E':xHEout, 'R':xHRout, 'A':xHAout}
    yHoutAll = {'E':yHEout, 'R':yHRout, 'A':yHAout}
    
    #Violations
    ptEviolate = numpy.any(yHEout[1:]-yHEout[:-1] < 0)
    if ptEviolate: print "Monotonic violation in E"
    ptRviolate = numpy.any(yHRout[1:]-yHRout[:-1] < 0)
    if ptRviolate: print "Monotonic violation in R"
    ptAviolate = numpy.any(yHAout[1:]-yHAout[:-1] < 0)
    if ptAviolate: print "Monotonic violation in A"
    drEviolate = numpy.any(dHE < 0)
    if drEviolate: print "Slope violation in E"
    drRviolate = numpy.any(dHR < 0)
    if drRviolate: print "Slope violation in R"
    drAviolate = numpy.any(dHA < 0)
    if drAviolate: print "Slope violation in A"

    print "Spline E: " + str(yHEout)
    print "Derivs E = " + str(dHE)
    print "Spline R: " + str(yHRout)
    print "Derivs R = " + str(dHR)
    print "Spline A: " + str(yHAout)
    print "Derivs A = " + str(dHA)
    new_basis = LinFunctions(method='HermiteGeneral', lam_range=lam_range, fullg_e=gE, fullg_r=gR, fullg_a=gA)
    if return_basis_only: return new_basis
    integrand, variance = sourcesys.inv_var_xform(const_matricies, new_basis, xHout, verbose=False)
    print variance
    if penalty > 0 and not returnpoints: print "Penalty: %f" % penalty
    print "--------------------------------------------------------------------------------------------"
    if returnpoints:
        return variance, integrand, xHoutAll, yHoutAll, new_basis
    else:
        return variance['natural'] + penalty 

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

if __name__=="__main__":
    #generate sample length
    #dhdlstart = 34 #Row where data starts
    #dhdlenergy = 1 #column where energy is
    #dhdlstates = 4 #Column where dhdl to other states starts, also coulmn for U0
    g_en_start = 19 #Row where data starts in g_energy output
    g_en_energy = 1 #Column where the energy is located
    niterations = len(open('lj10/prod/energy10_10.xvg','r').readlines()[g_en_start:])
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
    lamC12 = flamC12sqrt(epsi_samp_space, sig_samp_space)
    lamC6 = flamC6sqrt(epsi_samp_space, sig_samp_space)
    #Initial u_kln
    u_kln = numpy.zeros([nstates,nstates,niterations])
    sanity_kln = numpy.zeros([nstates,nstates,niterations])
    
    const_unaffected_matrix = numpy.zeros([nstates,niterations])
    const_Un_matrix = numpy.zeros([nstates,niterations])
    const_R0_matrix = numpy.zeros([nstates,niterations])
    const_R1_matrix = numpy.zeros([nstates,niterations])
    const_R_matrix = numpy.zeros([nstates,niterations])
    const_A0_matrix = numpy.zeros([nstates,niterations])
    const_A1_matrix = numpy.zeros([nstates,niterations])
    const_A_matrix = numpy.zeros([nstates,niterations])
    lam_range = linspace(0,1,nstates)
    #Read in the data for the unaffected state
    for k in xrange(nstates):
        print "Importing LJ = %02i" % k
        energy_dic = {'full':{}, 'rep':{}}
        energy_dic['null'] = open('lj%s/prod/energy%s_null.xvg' %(k,k),'r').readlines()[g_en_start:] #Read in the null energies (unaffected) of the K states
        for l in xrange(nstates):
            energy_dic['full']['%s'%l] = open('lj%s/prod/energy%s_%s.xvg' %(k,k,l),'r').readlines()[g_en_start:] #Read in the full energies for each state at KxL
            if l == 10 or l == 0:
                energy_dic['rep']['%s'%l] = open('lj%s/prod/energy%s_%s_rep.xvg' %(k,k,l),'r').readlines()[g_en_start:] #Read in the repulsive energies at 0, nstates-1, and K
        #Fill in matricies
        for n in xrange(niterations):
            #Unaffected state
            const_Un_matrix[k,n] = float(energy_dic['null'][n].split()[g_en_energy])
            #Isolate the data
            for l in xrange(nstates):
                u_kln[k,l,n] = float(energy_dic['full']['%s'%l][n].split()[g_en_energy]) #extract the kln energy, get the line, split the line, get the energy, convert to float, store
            #Repulsive terms: 
            #R0 = U_rep[k,k,n] + dhdl[k,0,n] - Un[k,n]
            const_R0_matrix[k,n] = float(energy_dic['rep']['%s'%(0)][n].split()[g_en_energy]) - const_Un_matrix[k,n]
            #R1 = U_rep[k,k,n] + dhdl[k,-1,n] - Un[k,n]
            const_R1_matrix[k,n] = float(energy_dic['rep']['%s'%(10)][n].split()[g_en_energy]) - const_Un_matrix[k,n]
            const_R_matrix[k,n] = const_R1_matrix[k,n] - const_R0_matrix[k,n]
            #Finish the total unaffected term
            #Total unaffected = const_Un + U0 = const_Un + (U_full[k,0,n] - const_Un) = U_full[k,0,n]
            const_unaffected_matrix[k,n] = u_kln[k,0,n]
            #Attractive term
            #u_A = U_full[k,n] - constR[k,n] - const_unaffected[k,n]
            const_A0_matrix[k,n] = u_kln[k,0,n] - const_R0_matrix[k,n] - const_Un_matrix[k,n]
            const_A1_matrix[k,n] = u_kln[k,10,n] - const_R1_matrix[k,n] - const_Un_matrix[k,n]
            const_A_matrix[k,n] = const_A1_matrix[k,n] - const_A0_matrix[k,n]
    #Sanity check
    for l in xrange(nstates):
        sanity_kln[:,l,:] = lamC12[l]*const_R_matrix + lamC6[l]*const_A_matrix + const_unaffected_matrix
    del_kln = numpy.abs(u_kln - sanity_kln)
    print "Max Delta: %f" % numpy.max(del_kln)
    #pdb.set_trace()
    ##################################################
    ############### END DATA INPUT ###################
    ##################################################
    #Create master uklns
    #Convert to dimless
    u_klnE = u_kln
    u_kln = u_kln * kjpermolTokT 
    const_R_matrix = const_R_matrix * kjpermolTokT 
    const_A_matrix = const_A_matrix * kjpermolTokT 
    const_unaffected_matrix = const_unaffected_matrix * kjpermolTokT
    includeRef = False
    if includeRef:
        offset=1
    else:
        offset=0
    Nallstates = Nparm + nstates + offset #1 to recreate the refstate each time
    sig_range = (spacing(sigStartSpace**3,sigEndSpace**3,Nparm))**(1.0/3)
    epsi_range = spacing(epsiStartSpace,epsiEndSpace,Nparm)
    epsi_plot_range = spacing(epsiStartSpace,epsi_max,Nparm)
    comp = ncdata('complex', '.', u_kln_input=u_kln, nequil=0000, save_equil_data=True, manual_subsample=True, compute_mbar=True)
    #(DeltaF_ij, dDeltaF_ij) = comp.mbar.getFreeEnergyDifferences(uncertainty_method='svd-ew')
    #printFreeEnergy(DeltaF_ij,dDeltaF_ij)
    #Reun from the subsampled data
    u_kln = u_kln[:,:,comp.retained_indices]
    const_R_matrix = const_R_matrix[:,comp.retained_indices]
    const_A_matrix = const_A_matrix[:,comp.retained_indices]
    const_unaffected_matrix = const_unaffected_matrix[:,comp.retained_indices]
    niterations = len(comp.retained_indices)
    Ref_state = 6 #Reference state of sampling to pull from
    #if not (os.path.isfile('es_freeEnergies%s.npz'%spacename) and graphsfromfile) or not (os.path.isfile('es_%s/N%iRef%iOff%iEpsi%i.npz' % (spacename, Nparm, Ref_state, offset, Nparm-1)) and savedata): #nand gate
    if not (os.path.isfile('es_freeEnergies%s.npz'%spacename) and graphsfromfile) or not (os.path.isfile('es_%s/ns%iN%iepsi%i.npz' % (spacename, nstates, Nparm, Nparm-1)) and savedata): #nand gate
        #Create numpy arrys
        DelF = numpy.zeros([Nparm, Nparm])
        dDelF = numpy.zeros([Nparm, Nparm])
        f_k = comp.mbar.f_k
        f_k_sub = numpy.zeros(Nallstates)
        f_k_sub[:nstates] = f_k
        N_k = comp.mbar.N_k
        N_k_sub = numpy.zeros(Nallstates, numpy.int32)
        N_k_sub[:nstates] = N_k
        #Populate energies
        run_start_time = time.time()
        number_of_iterations = Nparm
        for iepsi in xrange(Nparm):
            initial_time = time.time()
            iteration = iepsi + 1
            print "Epsi index: %i" % iepsi
            #Save data files
            if not (os.path.isfile('es_%s/ns%iN%iepsi%i.npz' % (spacename, nstates, Nparm, iepsi)) and savedata): #nand gate
                epsi = epsi_range[iepsi]
                #Create Sub matrix
                u_kln_sub = numpy.zeros([Nallstates,Nallstates,niterations])
                u_kln_sub[:nstates,:nstates,:] = u_kln
                #Rebuild the reference state
                if includeRef:
                    Repsi = epsi_samp_space[Ref_state]
                    Rsig = sig_samp_space[Ref_state]
                    u_kln_sub[:nstates,nstates,:] = flamC12sqrt(Repsi,Rsig)*const_R_matrix + flamC6sqrt(Repsi,Rsig)*const_A_matrix + const_unaffected_matrix
                for isig in xrange(Nparm):
                    sig = sig_range[isig]
                    u_kln_sub[:nstates,isig+nstates+offset,:] = flamC12sqrt(epsi,sig)*const_R_matrix + flamC6sqrt(epsi,sig)*const_A_matrix + const_unaffected_matrix
                mbar = MBAR(u_kln_sub, N_k_sub, initial_f_k=f_k_sub, verbose = False, method = 'adaptive')
                (DeltaF_ij, dDeltaF_ij) = mbar.getFreeEnergyDifferences(uncertainty_method='svd-ew')
                if savedata:
                    if not os.path.isdir('es_%s' % spacename):
                        os.makedirs('es_%s' % spacename) #Create folder
                    numpy.savez('es_%s/ns%iN%iepsi%i.npz' % (spacename, nstates, Nparm, iepsi), DeltaF_ij=DeltaF_ij, dDeltaF_ij=dDeltaF_ij) #Save file
                    #numpy.savez('es_%s/N%iRef%iOff%iEpsi%i.npz' % (spacename, Nparm, Ref_state, offset, iepsi), DeltaF_ij=DeltaF_ij, dDeltaF_ij=dDeltaF_ij) #Save file
            else:
                DeltaF_file = numpy.load('es_%s/ns%iN%iepsi%i.npz' % (spacename, nstates, Nparm, iepsi))
                #DeltaF_file = numpy.load('es_%s/N%iRef%iOff%iEpsi%i.npz' % (spacename, Nparm, Ref_state, offset, iepsi))
                DeltaF_ij = DeltaF_file['DeltaF_ij']
                dDeltaF_ij = DeltaF_file['dDeltaF_ij']
            #printFreeEnergy(DeltaF_ij, dDeltaF_ij)
            if includeRef:
                DelF[iepsi,:] = DeltaF_ij[nstates,nstates+offset:]
                dDelF[iepsi,:] = dDeltaF_ij[nstates,nstates+offset:]
            else:
                DelF[iepsi,:] = DeltaF_ij[Ref_state,nstates:]
                dDelF[iepsi,:] = dDeltaF_ij[Ref_state,nstates:]
            laptime = time.clock()
            # Show timing statistics. copied from Repex.py, copywrite John Chodera
            final_time = time.time()
            elapsed_time = final_time - initial_time
            estimated_time_remaining = (final_time - run_start_time) / (iteration) * (number_of_iterations - iteration)
            estimated_total_time = (final_time - run_start_time) / (iteration) * (number_of_iterations)
            estimated_finish_time = final_time + estimated_time_remaining
            print "Iteration took %.3f s." % elapsed_time
            print "Estimated completion in %s, at %s (consuming total wall clock time %s)." % (str(datetime.timedelta(seconds=estimated_time_remaining)), time.ctime(estimated_finish_time), str(datetime.timedelta(seconds=estimated_total_time)))
        numpy.savez('es_freeEnergies%s.npz'%spacename, free_energy=DelF, dfree_energy=dDelF)
    else:
        #if os.path.isfile('es_%s/N%iRef%iOff%iEpsi%i.npz' % (spacename, Nparm, Ref_state, offset, Nparm-1)) and savedata: #Pull data from 
        DelF = {}
        dDelF = {}
        for nstates in [11,12]:
            Sstate = str(nstates)
            if os.path.isfile('es_%s/ns%iN%iepsi%i.npz' % (spacename, nstates, Nparm, Nparm-1)) and savedata: #Pull data from 
                DelF[Sstate] = numpy.zeros([Nparm, Nparm])
                dDelF[Sstate] = numpy.zeros([Nparm, Nparm])
                for iepsi in xrange(Nparm):
                    #DeltaF_file = numpy.load('es_%s/N%iRef%iOff%iEpsi%i.npz' % (spacename, Nparm, Ref_state, offset, iepsi))
                    DeltaF_file = numpy.load('es_%s/ns%iN%iepsi%i.npz' % (spacename, nstates, Nparm, iepsi))
                    DeltaF_ij = DeltaF_file['DeltaF_ij']
                    dDeltaF_ij = DeltaF_file['dDeltaF_ij']
                    if includeRef:
                        DelF[Sstate][iepsi,:] = DeltaF_ij[nstates,nstates+1:]
                        dDelF[Sstate][iepsi,:] = dDeltaF_ij[nstates,nstates+1:]
                    else:
                        DelF[Sstate][iepsi,:] = DeltaF_ij[Ref_state,nstates:]
                        dDelF[Sstate][iepsi,:] = dDeltaF_ij[Ref_state,nstates:]
            else:
                figdata = numpy.load('es_freeEnergies%s.npz'%spacename)
                DelF = figdata['free_energy']
                dDelF = figdata['dfree_energy']
    #pdb.set_trace()
    ###############################################
    ####### START SPECIFIC FREE ENERGY CALC #######
    ###############################################
    #Mapping is for UAmethane, NEoPentane, and C60 in that order
    realname = ['UAm', 'NOP', 'C60', 'LJ6', 'null', 'LJ11']
    nreal = len(realname)
    realepsi = numpy.array([1.2301, 3.4941, 1.0372, 0.7600, 0, 0.8])
    realsig  = numpy.array([0.3730, 0.6150, 0.9452, 1.0170, 0, 0.3])
    anrealepsi = numpy.array([1.2301, 3.4941, 1.0372, 0.7600])
    anrealsig  = numpy.array([0.3730, 0.6150, 0.9452, 1.0170])
    u_kln_sub = numpy.zeros([nstates+nreal,nstates+nreal,niterations])
    u_kln_sub[:nstates,:nstates,:] = u_kln
    f_k = comp.mbar.f_k
    f_k_sub = numpy.zeros(nstates+nreal)
    f_k_sub[:nstates] = f_k
    N_k = comp.mbar.N_k
    N_k_sub = numpy.zeros(nstates+nreal, numpy.int32)
    N_k_sub[:nstates] = N_k
    for imol in xrange(nreal):
        #Save data files
        epsi = realepsi[imol]
        sig = realsig[imol]
        #Create Sub matrix
        u_kln_sub[:nstates,imol+nstates+offset,:] = flamC12sqrt(epsi,sig)*const_R_matrix + flamC6sqrt(epsi,sig)*const_A_matrix + const_unaffected_matrix
    mbar = MBAR(u_kln_sub, N_k_sub, initial_f_k=f_k_sub, verbose = False, method = 'adaptive')
    (realDeltaF_ij, realdDeltaF_ij) = mbar.getFreeEnergyDifferences(uncertainty_method='svd-ew')
    printFreeEnergy(realDeltaF_ij,realdDeltaF_ij)
    for imol in xrange(nreal):
        print "Free energy of %s relative to LJ6: %.3f +- %.3f kcal/mol with %i states" % (realname[imol], realDeltaF_ij[Ref_state,nstates+imol]*kjpermolTokcal/kjpermolTokT, realdDeltaF_ij[Ref_state,nstates+imol]*kjpermolTokcal/kjpermolTokT, nstates)
    print realepsi
    print realsig
    pdb.set_trace()
    ###############################################
    ######### END FREE ENERGY CALCULATIONS ########
    ###############################################
    #Set up a mask
    if masked:
        maDelF = ma.masked_where(numpy.fabs(DelF) > 200, DelF)
        madDelF = ma.masked_where(numpy.fabs(DelF) > 200, dDelF)
        orgDelF = DelF
        orgdDelF = dDelF
        DelF = maDelF
        dDelF = madDelF
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
        #ylabel = r'$\epsilon$ in kcal/mol'
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
    lam_C12 = C12map(lam_range)
    lam_C6 = C6map(lam_range)
        
    f = {}
    plots = {}
    #Plot the sigma and epsilon free energies
    #f,plots = plt.subplots(2,2,sharex=True, sharey=True, figsize=(11,6))
    #f,plots = plt.subplots(2,2,sharex=True, sharey=True, figsize=(11,6))
    f['11'],plots['11'] = plt.subplots(2,1,sharex=True)
    f['12'],plots['12'] = plt.subplots(2,1,sharex=True)
    Fplot = {'11':plots['11'][0],'12':plots['12'][0]}
    dFplot = {'11':plots['11'][1],'12':plots['12'][1]}
    #dFplot = {'11':plots[1,0],'12':plots[1,1]}
    print "Filling figures with data..."
    '''
    Observant readers will notice that DelF and dDelF are in dimensions of [epsi,sig] but I plot in sig,epsi. That said, the color map is CORRECT with this method... somehow. I questioned it once and then decided to come back to it at a later date.
    '''
    #Convert epsi to kcal
    epsi_range *= kjpermolTokcal
    epsi_samp_space *= kjpermolTokcal
    realepsi *= kjpermolTokcal
    anrealepsi *= kjpermolTokcal
    epsiEndSpace *= kjpermolTokcal
    epsiPlotStart *= kjpermolTokcal
    epsiPlotEnd *= kjpermolTokcal
    epsi_plot_range *= kjpermolTokcal

    imgFplot={}
    imgdFplot ={}
    divFplot = {}
    caxFplot = {}
    cFplot = {}
    divdFplot = {}
    caxdFplot = {}
    cdFplot = {}
    for nstates in ['11','12']:
        imgFplot[nstates] = Fplot[nstates].pcolormesh(sig_range**sig_factor,epsi_range,DelF[nstates]/kjpermolTokT*kjpermolTokcal)
        #Set the colorbar on the 12 state
        cvmin, cvmax = (-14.542572154421956, 8.6595207877425739)
        cdvmin, cdvmax = (3.1897634261829015e-05, 0.22292838017499619) 
        if nstates == '12' or True:
            divFplot[nstates] = mal(Fplot[nstates])
            caxFplot[nstates] = divFplot[nstates].append_axes('right', size='5%', pad=0.05)
            cFplot[nstates] = f[nstates].colorbar(imgFplot[nstates], cax=caxFplot[nstates])
            #set the minmax colorscales
            #print cFplot.get_clim()
            #cvmin, cvmax = (-21.971123537881027, 20.78176716595965) #These are the 11 nstate plots
            cFplot[nstates].set_clim(vmin=cvmin, vmax=cvmax)
            #Reduce the number of ticks
            #tick_locator = ticker.MaxNLocator(nbins=5)
            #cFplot[nstates].locator = tick_locator
            #cFplot[nstates].update_ticks()
        #Error plot
        imgdFplot[nstates] = dFplot[nstates].pcolormesh(sig_range**sig_factor,epsi_range,dDelF[nstates]/kjpermolTokT*kjpermolTokcal)
        if nstates == '12' or True:
            divdFplot[nstates] = mal(dFplot[nstates])
            caxdFplot[nstates] = divdFplot[nstates].append_axes('right', size='5%', pad=0.05)
            #Set the minmax colorscales
            #print imgdFplot.get_clim()
            #cdvmin, cdvmax = (0.00019094581786378227, 0.45022226894935008) #These are the 11 nstate plots
            #sys.exit(0)
            cdFplot[nstates] = f[nstates].colorbar(imgdFplot[nstates], cax=caxdFplot[nstates])
            #Reduce the number of ticks
            #dtick_locator = ticker.MaxNLocator(nbins=5)
            #cdFplot[nstates].locator = dtick_locator
            #cdFplot[nstates].update_ticks()
        imgdFplot[nstates].set_clim(vmin=cdvmin, vmax=cdvmax)
        #!!! Uncomment this to remake the title
        #f.suptitle(r'$\Delta G$ (top) and $\delta\Delta G$(bottom) for LJ Spheres' + '\n in units of kcal/mol')
        #Create the scatter sampled data
        for i in xrange(int(nstates)):
            epsi = epsi_samp_space[i]
            sig = sig_samp_space[i]
            if i == Ref_state:
                marker_color = 'k'
                marker_size  = 70
                marker_style = 'D'
            else:
                marker_color = 'k'
                marker_size  = 60
                marker_style = 'x'
            #if lam == 0 and spacing is logspace:
            #    lam = 10**(StartSpace-1)
            Fplot[nstates].scatter(sig**sig_factor,epsi, s=marker_size, c=marker_color, marker=marker_style)
            dFplot[nstates].scatter(sig**sig_factor,epsi, s=marker_size, c='w', marker=marker_style)
        if plotReal and Ref_state == 6 and nstates == '12':
                Fplot[nstates].scatter(realsig**sig_factor, realepsi, s=60, c='k', marker='+')
                dFplot[nstates].scatter(realsig**sig_factor, realepsi, s=60, c='w', marker='+')
        #plotlam = sampled_lam
        #if spacing is logspace and plotlam[0] == 0 and not plotC12_6:
        #    plotlam[0] = 10**(StartSpace-1)
        Fplot[nstates].plot(sig_range**sig_factor, epsi_plot_range, linewidth=2, color='k')
        dFplot[nstates].plot(sig_range**sig_factor, epsi_plot_range, linewidth=2, color='w')
        if annotatefig:
            if plotReal and nstates == '12':
                xyarrs = zip(anrealsig[:-1]**sig_factor, anrealepsi[:-1])
                antxt = "Chemically Realistic LJ Spheres"
                xoffset = 0.9
                realnamelong = ['UA Methane', 'Neopentane', r'C$_{60}$ Sphere']
                #Label the actual points
                bbox_def = dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
                Fplot[nstates].annotate(realnamelong[0], #String
                               (realsig[0], realepsi[0]), xycoords='data', #Point Where are we annotating at
                               xytext=(4, -18), textcoords='offset points', #Placement of text either absolute ('data') or relative ('offset points') to the xycoords
                               bbox=bbox_def) #style
                Fplot[nstates].annotate(realnamelong[1], #String
                               (realsig[1], realepsi[1]), xycoords='data', #Point Where are we annotating at
                               xytext=(-62, -20), textcoords='offset points', #Placement of text either absolute ('data') or relative ('offset points') to the xycoords
                               bbox=bbox_def) #style
                Fplot[nstates].annotate(realnamelong[2], #String
                               (realsig[2], realepsi[2]), xycoords='data', #Point Where are we annotating at
                               xytext=(6, 10), textcoords='offset points', #Placement of text either absolute ('data') or relative ('offset points') to the xycoords
                               bbox=bbox_def) #style
            else:
                xyarrs = [(anrealsig[-1]**sig_factor, anrealepsi[-1])]
                antxt = "Reference State"
                xoffset = 1
            if nstates == '11':
                my_annotate(Fplot[nstates],
                        antxt,
                        xy_arr=xyarrs, xycoords='data',
                        xytext=(sig_range.max()/2*xoffset, epsiEndSpace/2), textcoords='data',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                        arrowprops=dict(arrowstyle="-|>",
                                        connectionstyle="arc3,rad=0.2",
                                        fc="w", linewidth=2))
            if nstates == '12':
                xypt = [(sig_samp_space[-1], epsi_samp_space[-1])]
                my_annotate(Fplot[nstates],
                            #"Extra Sampling",
                            "Additional Sampling",
                            xy_arr=xypt, xycoords='data',
                            xytext=(sigPlotStart*1.05, epsiEndSpace/2*1.05), textcoords='data',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                            arrowprops=dict(arrowstyle="-|>",
                                            connectionstyle="arc3,rad=0.2",
                                            fc="w", linewidth=2))
        for ax in [Fplot[nstates],dFplot[nstates]]:
            ax.set_yscale(spacename)
            ax.set_xscale(spacename)
            if nstates == '11' or True:
                ax.set_ylim([epsiPlotStart,epsiPlotEnd])
            ax.set_xlim([sigPlotStart,sigPlotEnd])
            ax.patch.set_color('grey')
            #Reduce the number of ticks
            ax.locator_params(nbins=5)
        #dFplot[nstates].set_xlabel(xlabel, fontsize=20)
        #Fplot[nstates].set_title('Number of\nsampled states: %i'%int(nstates),fontsize=15)
        f[nstates].subplots_adjust(hspace=0.02, wspace=0.02, top=.91, left=.1, right=.88)
        f[nstates].text(0.04, .5, ylabel, rotation='vertical', horizontalalignment='center', verticalalignment='center', fontsize=21)
        f[nstates].text(0.5, .04, xlabel,  horizontalalignment='center', verticalalignment='center', fontsize=21)
        #Name of each plot
        f[nstates].text(0.965, .71, r'$\Delta G$',  rotation=-90, horizontalalignment='center', verticalalignment='center', fontsize=21)
        f[nstates].text(0.965, .27, r'$\delta\left(\Delta G\right)$', rotation=-90, horizontalalignment='center', verticalalignment='center', fontsize=21)
    #Fplot['11'].set_title('11 Sampled States')
    #Fplot['12'].set_title('Naive additional sampling at 12 states')
        if savefigs:
            f[nstates].patch.set_alpha(0.0)
            f[nstates].savefig('LJParameterization_nstates%s.png' % nstates, bbox_inches='tight', dpi=600)  
    
    #if savefigs:
    #    print "Rendering figures..."
    #    if plotReal:
    #        plotrealstr = "T"
    #    else:
    #        plotrealstr = "F"
    #    f.patch.set_alpha(0.0)
    #    #f.savefig('LJ_GdG_ns%i_es%i_real%s_N%i_em%2d.png' % (nstates, sig_factor, plotrealstr, Nparm, epsiEndSpace*10), bbox_inches='tight', dpi=600)  
    #    #f.savefig('LJ_GdG_ns%i_es%i_real%s_N%i_em%2d.eps' % (nstates, sig_factor, plotrealstr, Nparm, epsiEndSpace*10), bbox_inches='tight', dpi=600)
    #    f.savefig('LJParameterizationComparison.png', bbox_inches='tight', dpi=600)  
    #    #f.savefig('LJParameterizationComparison.eps', bbox_inches='tight')  
    #    #f.savefig('LJ_GdG_ns%i_es%i_real%s_N%i_em%1.1f.eps' % (nstates, sig_factor, plotrealstr, Nparm, epsiEndSpace), bbox_inches='tight')  
    #    #f.savefig('LJ_GdG_ns%i_es%i_real%s_N%i_em%1.1f.pdf' % (nstates, sig_factor, plotrealstr, Nparm, epsiEndSpace), bbox_inches='tight')  
    #    #f.savefig('LJ_GdG_ns%i_es%i_real%s_N%i_em%1.1f.eps' % (nstates, sig_factor, plotrealstr, Nparm, epsiEndSpace), bbox_inches='tight')  
    #else:
    #    plt.show()
    #pdb.set_trace()
    sys.exit(0) #Terminate here
####################################################################################
####################################################################################
####################################################################################
    const_matricies = {'R':const_R_matrix, 'A':const_A_matrix, 'E':const_E_matrix, 'P':const_P_matrix, 'Un':const_Un_matrix, }
    
    sampled_lam = linspace(0,1,21)
    lam_range = linspace(0,1,101)
    
    truncate = False
    print "Loading Complex..."
    #comp = ncdata('complex', '.', u_kln_input=u_kln, nequil=0000, save_equil_data=True, manual_subsample=True, compute_mbar=True)
    print comp.u_kln.shape
    vac = comp
    vacS = compS
    basis = LinFunctions(method='PureLin')
    sysX = BasisVariance(basis, comp, vac)
    sysS = BasisVariance(basis, compS, vacS)
    if not (os.path.isfile('hbr-all.npz') and graphsfromfile):
        integrandSamp, varianceSamp, dhdlSamp = sysX.vargenerate_xform(const_matricies,calculatedhdl=True, bootstrap_error=True, bootstrap_count=2, lam_in_r=sampled_lam, lam_in_a=sampled_lam, lam_in_e=sampled_lam)
        integrand, variance, dhdl = sysX.vargenerate_xform(const_matricies,calculatedhdl=True, bootstrap_error=True, bootstrap_count=2)
        integrandShort, varianceShort, dhdlShort = sysS.vargenerate_xform(Sconst_matricies,calculatedhdl=True, bootstrap_error=True, bootstrap_count=2)
        integrandSampShort, varianceSampShort, dhdlSampShort = sysS.vargenerate_xform(Sconst_matricies,calculatedhdl=True, bootstrap_error=True, bootstrap_count=2, lam_in_r=lam_short, lam_in_a=lam_short, lam_in_e=lam_short)
        collection = [integrand,variance, integrandSamp, varianceSamp, integrandShort, varianceShort, integrandSampShort,varianceSampShort]
        for item in collection:
            for key in item.keys():
                item[key] *= comp.kcalmolsq
        for dhdldic in [dhdl, dhdlSamp, dhdlShort, dhdlSampShort]:
            for key in dhdl.keys():
                dhdl[key] *= comp.kcalmol
        if graphsfromfile:
            numpy.savez('hbr-all.npz', integrand=integrand, variance=variance, dhdl=dhdl)
    else:
        imageArrays = numpy.load('hbr-all.npz')
        integrand = imageArrays['integrand'].item()
        variance = imageArrays['variance'].item()
        dhdl = imageArrays['dhdl'].item()
    
    
    x = linspace(0,1,101)
    xSamp = linspace(0,1,21)
    xSampShort = linspace(0,1,11)
    lam_range=linspace(0,1,101)
    y = integrand['natural']
    ySamp = integrandSamp['natural']
    ySampShort = integrandSampShort['natural']
    yShort = integrandShort['natural']
    yp = integrand['plus']  
    ym = integrand['minus'] 
    z = dhdl['natural']     
    zp = dhdl['plus']       
    zm = dhdl['minus']      
    print "Variance: %f" % (variance['natural'])
    print "Variance kT: %f" % (variance['natural']/comp.kcalmolsq)
    
    #f,(varorig,pltorigbasis,varopti,pltoptibasis) = plt.subplots(4,1)
    f,(varorig) = plt.subplots(1,1)
    origvarline = varorig.plot(x,y, '-b', label='Var orig')
    #varorig.plot(x,yp,'--b')
    #varorig.plot(x,ym,'--b')
    varorig.scatter(xSamp, ySamp, marker='x', s=50, c='b')
    varorig.plot(x,yShort, '-r', label='Var short')
    varorig.scatter(xSampShort, ySampShort, marker='d', s=50, c='r')
    varorig.set_ylabel('Variance, Original')
    varorig.set_xlim([0,1])
    varorig.set_ylim([0,2000])
    dhdlorig = varorig.twinx()
    origdhdlline = dhdlorig.plot(x,z,'-g', label='Dhdl orig')
    dhdlorig.plot(x,zp,'--g')
    dhdlorig.plot(x,zm,'--g')
    dhdlorig.set_ylabel('Dhdl')
    dhdlorig.set_xlim([0,1])
    origlines = origvarline + origdhdlline
    origlinelabels = [l.get_label() for l in origlines]
    varorig.legend(origlines, origlinelabels, loc='upper center')
    f.suptitle('hbr')
    
    plt.show()
    pdb.set_trace()
    
    ################# Optimization routine ######################
    Npoints = 11 #Number of fitting points in new basis set; to be uniformly distributed
    x_g = scipy.linspace(0,1,Npoints)
    initial_Eg = x_g.copy()
    initial_Rg = x_g.copy()
    initial_Ag = x_g.copy()
    guess_Eg = initial_Eg[1:-1]
    guess_Rg = initial_Rg[1:-1]
    guess_Ag = initial_Ag[1:-1]
    Ng = len(guess_Eg)
    guess_gs = numpy.concatenate((guess_Eg,guess_Rg,guess_Ag))
    #Constrain points to ensure adjacent points are monotonic
    constraints=[]
    penaltytol=.01
    for i in range(len(guess_gs)): #No point less than 0 or greater than 1, also convers the previous 2 constrants
        constraints.append(lambda x,*args: (x[i]-penaltytol)*1000)
        constraints.append(lambda x,*args: (1 - x[i]-penaltytol)*1000)
    #Eg constraints
    for i in range(0*Ng+1, 1*Ng):
        constraints.append(lambda x,*args: (x[i] - x[i-1]- penaltytol)*1000)
    for i in range(1*Ng-1):
        constraints.append(lambda x,*args: (x[i+1] - x[i]- penaltytol)*1000)
    #Rg constraints
    for i in range(1*Ng+1, 2*Ng):
        constraints.append(lambda x,*args: (x[i] - x[i-1]- penaltytol)*1000)
    for i in range(2*Ng-1):
        constraints.append(lambda x,*args: (x[i+1] - x[i]- penaltytol)*1000)
    #Ag constraints
    for i in range(2*Ng+1, 3*Ng):
        constraints.append(lambda x,*args: (x[i] - x[i-1]- penaltytol)*1000)
    for i in range(3*Ng-1):
        constraints.append(lambda x,*args: (x[i+1] - x[i]- penaltytol)*1000)
    
    startrho = (initial_Eg[1] - initial_Eg[0])/1.5
    returnpts = False
    method = 'hermite'
    constructorsys = sysX
    if not constructorsys.complex.mbar_ready: 
        print "Computing f_k for MBAR before run"
        constructorsys.complex.compute_mbar()
    junkbasis = LinFunctions(method='LinA', hrconstant=100)
    optimized_g = numpy.concatenate((guess_Eg, guess_Rg, guess_Ag))
    #optimized_g = scipy.optimize.fmin_cobyla(generalvar, guess_gs, constraints, args=(x_g, Ng, const_matricies, constructorsys, returnpts, method, penaltytol), consargs=(), rhobeg = startrho, maxfun = 10000)
    optimized_g=numpy.array([ 
        0.17238638, 0.35225791, 0.36201698, 0.39133392, 0.5332047, 0.58403442, 0.59397454, 0.81362049, 0.99079778, 
        0.18966892, 0.35518083, 0.37440938, 0.4442811, 0.54613138, 0.65536611, 0.66541912, 0.73401692, 0.78753168, 
        0.08158787, 0.19779917, 0.30786773, 0.45985335, 0.47253737, 0.59097645, 0.7458982, 0.828316, 0.9224944])
    print 'final_g:' + str(optimized_g)
    
    optimized_basis = generalvar(optimized_g, x_g, Ng, const_matricies, constructorsys, return_basis_only=True)
    if not (os.path.isfile('hbr-allOpti.npz') and graphsfromfile): #nand gate
        ointegrand, ovariance, odhdl = constructorsys.inv_var_xform(const_matricies,optimized_basis, lam_range, verbose=False, calculatedhdl=True, bootstrap_error=True, bootstrap_count=2)
        ocollection = [ointegrand,ovariance]
        for item in ocollection:
            for key in item.keys():
                item[key] *= comp.kcalmolsq
        for key in odhdl.keys():
            odhdl[key] *= comp.kcalmol
        print "Variance: %f" % ovariance['natural']
    else:
        imageArraysOpti = numpy.load('hbr-allOpti.npz')
        ointegrand = imageArraysOpti['integrand'].item()
        ovariance = imageArraysOpti['variance'].item()
        odhdl = imageArraysOpti['dhdl'].item()
    if graphsfromfile and not os.path.isfile('hbr-allOpti.npz'):
        numpy.savez('hbr-allOpti.npz', integrand=ointegrand, variance=ovariance, dhdl=odhdl)
    
    print "Variance: %f" % (ovariance['natural'])
    
    oy = ointegrand['natural']
    oyp = ointegrand['plus']
    oym = ointegrand['minus']
    oz = odhdl['natural']
    ozp = odhdl['plus']
    ozm = odhdl['minus']
    
    optivarline = varopti.plot(x,oy,'-r', label="Var, Opti")
    varopti.plot(x,oyp,'--r')
    varopti.plot(x,oym,'--r')
    dhdlopti = varopti.twinx()
    optidhdlline = dhdlopti.plot(x,oz,'-m', label="Dhdl Opti")
    dhdlopti.plot(x,ozp,'--m')
    dhdlopti.plot(x,ozm,'--m')
    dhdlopti.set_ylabel('Dhdl')
    varopti.set_ylabel('Variance, Optimal')
    optilines = optivarline + optidhdlline
    optilinelabels = [l.get_label() for l in optilines]
    varopti.legend(optilines, optilinelabels, loc='upper center')
    
    pltorigbasis.plot(x,basis.h_e(x), 'b-', label='Eorig')
    pltorigbasis.plot(x,basis.h_r(x), 'b--', label='Rorig')
    pltorigbasis.plot(x,basis.h_a(x), 'b+', label='Aorig')
    pltoptibasis.plot(x,optimized_basis.h_e(x), 'r-', label='Eopti')
    pltoptibasis.plot(x,optimized_basis.h_r(x), 'r--', label='Ropti')
    pltoptibasis.plot(x,optimized_basis.h_a(x), 'r+', label='Aopti')
    pltorigbasis.legend(loc='upper left')
    pltoptibasis.legend(loc='upper left')
    
    varopti.set_ylim([0,5000])
    varorig.set_ylim([0,5000])
    
    pdb.set_trace()
    plt.show()
