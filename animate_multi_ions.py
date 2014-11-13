from esq_construct_ukln import *
import matplotlib.animation as ani

Ref_state = 1

#Adjust this list to get your image, figure dynamicly updates
nstates = [21, 31, 41, 51, 61]
nstates = [47, 53]
nstates = [21,69]

def animate(q_samp_space, epsi_samp_space, sig_samp_space):
    epsi_min = epsi_samp_space[0]
    epsi_max = epsi_samp_space[5]
    sig_min = sig_samp_space[0]
    sig_max = sig_samp_space[5]
    q_min = -2.0
    q_max = +2.0

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

    sig_range = (spacing(sigStartSpace**3,sigEndSpace**3,Nparm))**(1.0/3)
    epsi_range = spacing(epsiStartSpace,epsiEndSpace,Nparm)
    q_range = spacing(qStartSpace,qEndSpace,Nparm)
    epsi_plot_range = spacing(epsiStartSpace,epsi_max,Nparm)
    #Go through all nstates
    nplot = len(nstates)
    DelF = {}
    dDelF={}
    reldDelF={}
    maxerr = 0
    minerr = 1E10
    #Import all data
    print 'Importing Data...'
    for istates in nstates:
        state = str(istates)
        DelF[state] = numpy.zeros([Nparm, Nparm, Nparm])
        dDelF[state] = numpy.zeros([Nparm, Nparm, Nparm])
        for iq in xrange(Nparm):
            sys.stdout.flush()
            sys.stdout.write('\rLoading State %i, loading file %d/%d...' % (istates, iq,Nparm-1))
            DeltaF_file = numpy.load('esq_%s/ns%iNp%iQ%i.npz' % (spacename, istates, Nparm, iq))
            DeltaF_ij = DeltaF_file['DeltaF_ij']
            dDeltaF_ij = DeltaF_file['dDeltaF_ij']
            for iepsi in xrange(Nparm):
                #if includeRef:
                #    DelF[iq, iepsi,:] = DeltaF_ij[nstates,nstates+1:]
                #    dDelF[iq, iepsi,:] = dDeltaF_ij[nstates,nstates+1:]
                #Unwrap the data
                DelF[state][iq, iepsi,:] = DeltaF_ij[Ref_state, iepsi*Nparm:(iepsi+1)*Nparm]
                dDelF[state][iq, iepsi,:] = dDeltaF_ij[Ref_state, iepsi*Nparm:(iepsi+1)*Nparm]
        sys.stdout.write('\n')
        reldDelF[state] = numpy.abs(dDelF[state]/DelF[state])
        if numpy.nanmin(dDelF[state]) < minerr:
            minerr = numpy.nanmin(dDelF[state])
        if numpy.nanmax(dDelF[state]) > maxerr:
            maxerr = numpy.nanmax(dDelF[state])
    #Clean NaN
    for istates in nstates:
        state=str(istates)
        dDelF[state][numpy.isnan(dDelF[state])] = maxerr

    
    print 'Initilizing figures...'
    xlabel = r'$\sigma$ in nm'
    ylabel = r'$\epsilon$ in kJ/mol'
    
    relativeErr = False
    fixErr = False
    if relativeErr:
        f,(daplots) = plt.subplots(3,nplot,sharex=True, sharey=True)
        rdFig = f
        print "Relative error Not yet implemented"
        sys.exit(1)
    else:
        f,(daplots) = plt.subplots(2,nplot,sharex=True, sharey=True)
        g,rdFplot = plt.subplots(1,nplot)
        rdFig = g
    if fixErr:
        errorlims = numpy.load('n24_error_lims.npy')
        cdvmin = errorlims[0]
        cdvmax = errorlims[1]
        errstr='ErrFix'
    else:
        cdvmin = minerr*1.01
        cdvmax = maxerr*1.01
        errstr='ErrVar'
    #Initilize figures
    imgFplot={}
    imgdFplot={}
    for i in xrange(nplot):
        state = str(nstates[i])
        ### Main plot ###
        if nplot == 1:
            imgFplot[state] = daplots[0].pcolormesh(sig_range, epsi_range, DelF[state][(Nparm-1)/2,:,:])
        else:
            imgFplot[state] = daplots[0,i].pcolormesh(sig_range, epsi_range, DelF[state][(Nparm-1)/2,:,:])
        if i == nplot-1:
            if nplot == 1:
                divFplot = mal(daplots[0])
            else:
                divFplot = mal(daplots[0,i])
            caxFplot = divFplot.append_axes('right', size='5%', pad=0.05)
            cFplot = f.colorbar(imgFplot[state],cax=caxFplot)
        ### Error Plot ###
        if nplot == 1:
            imgdFplot[state] = daplots[1].pcolormesh(sig_range, epsi_range, dDelF[state][(Nparm-1)/2,:,:])
        else:
            imgdFplot[state] = daplots[1,i].pcolormesh(sig_range, epsi_range, dDelF[state][(Nparm-1)/2,:,:])
        imgdFplot[state].set_clim(vmin=cdvmin, vmax=cdvmax)
        if i == nplot-1:
            if nplot == 1:
                divdFplot = mal(daplots[1])
            else:
                divdFplot = mal(daplots[1,i])
            caxdFplot = divdFplot.append_axes('right', size='5%', pad=0.05)
            cdFplot = f.colorbar(imgdFplot[state],cax=caxdFplot)
        #Format the static objects
        sup_title_template = r'$\Delta G$ (top) and $\delta\Delta G$(bottom) with $q=%.2f$ for ions' + '\n in units of kcal/mol'
        indi_title_template=r'%d sampled states'
        titlefontsize = 12
        ftitle = f.suptitle('', fontsize = titlefontsize)
        for ax in daplots.ravel():
            ax.set_yscale(spacename)
            ax.set_xscale(spacename)
            ax.set_ylim([epsiPlotStart,epsiPlotEnd])
            ax.set_xlim([sigPlotStart,sigPlotEnd])
            ax.patch.set_color('grey')
        for i in xrange(nplot):
            state = nstates[i]
            if nplot ==1:
                #daplots[0].set_title(indi_title_template % state, fontsize = titlefontsize-1)
                pass
            else:
                daplots[0,i].set_title(indi_title_template % state, fontsize = titlefontsize-1)
        f.subplots_adjust(hspace=0.02, wspace=0.05)
        f.text(0.05, .5, ylabel, rotation='vertical', horizontalalignment='center', verticalalignment='center', fontsize=18)
        f.text(0.5, .01, xlabel, horizontalalignment='center', verticalalignment='center', fontsize=18)
        #Animate those figures!
        def cleanup():
            returnlist = []
            for key in imgFplot.keys():
                imgFplot[key].set_array(numpy.array([]))
                imgdFplot[key].set_array(numpy.array([]))
                returnlist.append(imgFplot[key])
                returnlist.append(imgdFplot[key])
            ftitle.set_text('')
            returnlist.append(ftitle)
            #Best way I can find to expand the dictionary to return the items
            return tuple(returnlist)
        def moveq(qndx):
            q = q_range[qndx]
            scrapfig, dascrap = plt.subplots(2,nplot)
            scrapFplot = {}
            scrapdFplot = {}
            cmax = -1E5
            cmin = 1E5
            #Generate scrap figures
            for i in xrange(nplot):
                state = str(nstates[i])
                if nplot == 1:
                    scrapFplot[state] = dascrap[0].pcolormesh(sig_range**sig_factor,epsi_range,DelF[state][qndx,:,:])
                else:
                    scrapFplot[state] = dascrap[0,i].pcolormesh(sig_range**sig_factor,epsi_range,DelF[state][qndx,:,:])
                (curmin, curmax) = scrapFplot[state].get_clim()
                #Determine limts on the DelF plot
                if curmin < cmin:
                    cmin = curmin
                if curmax > cmax:
                    cmax = curmax
                if nplot == 1:
                    scrapdFplot[state] = dascrap[1].pcolormesh(sig_range**sig_factor,epsi_range,dDelF[state][qndx,:,:], vmax=cdvmax, vmin=cdvmin)
                else:
                    scrapdFplot[state] = dascrap[1,i].pcolormesh(sig_range**sig_factor,epsi_range,dDelF[state][qndx,:,:], vmax=cdvmax, vmin=cdvmin)
            #Now that limts are known, apply them
            for i in xrange(nplot):
                state = str(nstates[i])
                imgFplot[state].set_array(scrapFplot[state].get_array())
                imgFplot[state].set_clim(vmin=cmin, vmax=cmax)
                imgdFplot[state].set_array(scrapdFplot[state].get_array())
            #Collapse scrap figure
            plt.close(scrapfig)
            returnlist = []
            for key in imgFplot.keys():
                returnlist.append(imgFplot[key])
                returnlist.append(imgdFplot[key])
            ftitle.set_text(sup_title_template % q)
            returnlist.append(ftitle)
            return tuple(returnlist)
    
    #Call animation and save
    print 'Animating figure...'
    statenames = ''
    for state in nstates:
        statenames+=str(state) + '_'
    filename = 'Animated_Charge_%s_%s.mp4' % (statenames, errstr)
    aniU = ani.FuncAnimation(f, moveq, range(Nparm), interval=150, blit=False, init_func=cleanup)
    #pdb.set_trace()
    aniU.save(filename, dpi=400)

if __name__=="__main__":
    parms = numpy.load('qes.npy')
    qs = parms[:,0]
    es = parms[:,1]
    ss = parms[:,2]
    animate(qs, es, ss)
