import numpy
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import scipy.fftpack
from scipy import linspace
import scipy.stats as stats
from scipy.interpolate import splprep,splrep,splev
from scipy.interpolate import UnivariateSpline as US
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit
import gen_rdf_data as genrdf
import pdb
import os.path
from numpy.random import random_integers

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')



ions = ['Li+', 'Cl-']
ions = ['Li+',      'Na+',       'K+',        'Rb+',       'Cs+',       'F-',        'Cl-',       'Br-',       'I-']
ionsshort=['li',     'na',        'k',         'rb',        'cs',        'f',         'cl',        'br',        'i' ]
nions = len(ions)
nstates = 203
filestring = 'pertrdfs/pertrdfOn%s%s.npz'
samplefile = 'lj%s/prod/rdfOhist%sb160.npz'
basebins = genrdf.defbin

savefig = True
sampled = True
bootstrap = True
availablesampled = ionsshort
fade = False
fadedist = 0.8
fit = False

Erdfs = {}
dErdfs = {}
figs = {}
plots = {}

plotx = linspace(genrdf.distmin, genrdf.distmax, basebins)/10.0 #put in nm

def simplefit(x, M, L, A, B, T, xprep=None, yprep=None):
    '''
    1 + y^-m[g(d) -1-lam] + [(y-1+lam)/y]*exp[-A(y-1)]cos[B(y-1)] for m >= 1 and y>=1
    g(d)exp(-T(y-1)^2) for y<1
    y=r/d
    d=Emax
    Params: m, lam, A, B T
    '''
    if xprep is None or yprep is None:
        raise
    else:
        gd = yprep.max()
        d = xprep[numpy.where(yprep==gd)]
        
    y=x/d
    g = numpy.zeros(x.shape)
    g[y<1] = gd*numpy.exp(-T*(y[y<1]-1)**2)
    g[y>=1] = 1 + y[y>=1]**(-numpy.abs(M))*(gd-1-L) + ((y[y>=1]-1+L)/y[y>=1])*numpy.exp(-A*(y[y>=1]-1))*numpy.cos(B*(y[y>=1]-1))
    return g

def ffade(x, g, fadedist=fadedist):
    #Fade function
    loc = numpy.where(x >= fadedist)
    g[loc] = 1 + (g[loc] -1)*numpy.exp(-(x[loc]/fadedist-1)**2)
    return g

def fftxform(x,y):
    N = x.size
    w = scipy.fftpack.rfft(y)
    f = scipy.fftpack.rfftfreq(N, x[1]-x[0])
    spectrum = w**2
    
    cutoff_idx = spectrum < (spectrum.max()/5)
    w2 = w.copy()
    w2[cutoff_idx] = 0
    
    y2 = scipy.fftpack.irfft(w2)
    return y2

def weighted_moving_average(x,y,step_size=0.05,width=1):
    bin_centers  = numpy.arange(numpy.min(x),numpy.max(x)-0.5*step_size,step_size)+0.5*step_size
    bin_avg = numpy.zeros(len(bin_centers))

    #We're going to weight with a Gaussian function
    def gaussian(x,amp=1,mean=0,sigma=1):
        return amp*numpy.exp(-(x-mean)**2/(2*sigma**2))

    for index in range(0,len(bin_centers)):
        bin_center = bin_centers[index]
        weights = gaussian(x,mean=bin_center,sigma=width)
        bin_avg[index] = numpy.average(y,weights=weights)

    return (bin_centers,bin_avg)

for l in xrange(nions):
    ion = ions[l]
    shortion = ionsshort[l]
    rdfdata = numpy.load(filestring % (nstates, ion))
    Erdfs[ion] = rdfdata['Erdfs']
    dErdfs[ion] = rdfdata['dErdfs']
    figs[ion], plots[ion] = plt.subplots(1,1)
    p = plots[ion]
    f = figs[ion]
    dx = plotx[1] - plotx[0]
    gausrdf = ndimage.filters.gaussian_filter(Erdfs[ion], (plotx[-1]-plotx[0]), truncate=4.0)
    gausrdf2 = ndimage.filters.gaussian_filter(Erdfs[ion], (plotx[-1]-plotx[0]), truncate=1.0)
    gausrdf3 = ndimage.fourier.fourier_gaussian(Erdfs[ion], plotx[20]-plotx[0])
    #pdb.set_trace()
    #gx, gy = weighted_moving_average(plotx, Erdfs[ion], step_size=dx*.5, width=dx*1)
    #p.plot(gx,gy, '-r', linewidth=2)
    #pol = numpy.polyfit(plotx, Erdfs[ion], 50)
    #w = numpy.exp(-plotx/plotx.max())
    #w = 1.0/dErdfs[ion]
    #w[numpy.isnan(w)] = 0.0
    #w[numpy.isinf(w)] = w[numpy.logical_not(numpy.isinf(w))].max()
    #rdfus = US(plotx, Erdfs[ion], k=3, w=w)
    #tck = scipy.interpolate.splrep(plotx,Erdfs[ion], k=3, s=None, w=w)
    #interp1d = scipy.interpolate.interp1d(plotx, Erdfs[ion], kind='cubic')
    #splrdf = scipy.interpolate.splev(plotx,tck)
    #fftrdf = fftxform(plotx, Erdfs[ion])
    #sgrdf = savitzky_golay(Erdfs[ion], 21, 3)
    #rbfrdf = scipy.interpolate.LSQUnivariateSpline(plotx, Erdfs[ion], plotx[1:-1:3])(plotx)
    if fade:
        Erdfs[ion] = ffade(plotx, Erdfs[ion])
    if fit:
        fitfunc = lambda x, M, L, A, B, T: simplefit(x, M, L, A, B, T, xprep=plotx, yprep=Erdfs[ion])
        pdb.set_trace()
        stdev = dErdfs[ion].copy()
        #stdev[numpy.isnan(stdev)] = stdev[numpy.logical_not(numpy.isinf(stdev))].max()
        stdev[numpy.isnan(stdev)] = 0
        stdev[stdev==0] = stdev[stdev!=0].min()
        try:
            popt, pvoc = curve_fit(fitfunc, plotx, Erdfs[ion], sigma=stdev, absolute_sigma=True)
        except RuntimeError:
            try:
                popt, pvoc = curve_fit(fitfunc, plotx, Erdfs[ion], sigma=stdev, absolute_sigma=True, maxfev=1000*6)
            except:
                popt, pvoc = curve_fit(fitfunc, plotx, Erdfs[ion], maxfev=1000*6)
        fity = fitfunc(plotx, *popt)
        p.plot(plotx, fity, '-r', linewidth=2, label="Fit Function")
    #tck = splrep(plotx, Erdfs[ion], s=5)
    #p.scatter(plotx, Erdfs[ion], s=15, facecolors='none', edgecolors='k', label='MBAR Estimated')
    #p.plot(plotx, gausrdf, '-r', linewidth=2, label="Gaussian Filter of MBAR")
    #p.plot(plotx, numpy.polyval(pol, plotx), '-b', linewidth=2)
    #p.plot(plotx, splrdf, '-b', linewidth=2)
    #p.plot(plotx, rdfus(plotx), '--g', linewidth=2)
    #p.plot(plotx, rbfrdf, '-k', linewidth=1)
    #p.plot(plotx, splev(plotx, tck, der=0), '-k')
    #p.plot(plotx, rdfus(plotx), '-g', linewidth=2)
    p.plot(plotx, Erdfs[ion], '-k', linewidth = 2, marker='o', markeredgecolor='k', ms=0, label="MBAR Estimated")
    p.plot(plotx, Erdfs[ion]+2*dErdfs[ion], '--k', linewidth = 1)
    p.plot(plotx, Erdfs[ion]-2*dErdfs[ion], '--k', linewidth = 1)
    if sampled and (shortion in availablesampled):
        sampledrdffile = numpy.load(samplefile % (shortion, shortion))
        samplerdf = sampledrdffile['rdf']
        samplerange = sampledrdffile['histrange']
        sx = linspace(samplerange[0], samplerange[1], 160)/10.0
        sy = samplerdf.sum(axis=1)/float(samplerdf.shape[1])
        if fade:
            sy = ffade(sx,sy)
        p.plot(sx, sy, '-g', linewidth=2, label="Direct Simulation of Ion")
        if bootstrap:
            nboot = 200
            rdfboot = numpy.zeros((nboot, samplerdf.shape[0], samplerdf.shape[1]))
            for iboot in xrange(nboot):
                samplepool = random_integers(0,samplerdf.shape[1]-1,samplerdf.shape[1])
                rdfboot[iboot,:,:] = samplerdf[:,samplepool]
            bsy = rdfboot.sum(axis=-1)/float(rdfboot.shape[-1])
            dsy = numpy.sqrt(numpy.var(bsy,axis=0))
            p.plot(sx,sy+2*dsy, '--g', linewidth=1)
            p.plot(sx,sy-2*dsy, '--g', linewidth=1)
        
    p.set_ylim([0,plots[ion].get_ylim()[1]])
    p.set_xlim([0.0, 1.2])
    p.set_xlabel(r'$r$ in nm', fontsize=15)
    p.set_ylabel(r'$g(r)_{\mathrm{Ion-O}}$', fontsize=17)
    p.legend(loc='upper right')
    p.set_title(ion)
    if savefig:
        f.patch.set_alpha(0)
        p.patch.set_alpha(1)
        p.patch.set_color('w')
        f.savefig('rdf%s.eps'%ion, bbox_inches='tight')
        try:
            os.mkdir('rdfpngs')
        except:
            pass
        f.savefig('rdfpngs/rdf%s.png'%ion, bbox_inches='tight')
    #Print the rdf=0 location
    #Make a blank array of 0 for comparison with the isclose command
    blank0 = numpy.zeros(Erdfs[ion].shape)
    #x0max = plotx[numpy.where(Erdfs[ion]==0)].max()
    x0max = plotx[numpy.isclose(Erdfs[ion], blank0)].max()
    print "Max location where %s RDF is 0: r=%f" % (ion, x0max)
    if bootstrap:
        meandMbar = numpy.nanmean(dErdfs[ion])
        meandBoot = numpy.abs(dsy).mean()
        print "Mean Error: MBAR=%f -- Sim Bootsrap=%f" % (meandMbar, meandBoot)
        print "MBAR is %f %% on average of Simulation Bootstrap" % (meandMbar/meandBoot * 100)
    #plt.show()
if not savefig:
    plt.show()
    pdb.set_trace()
