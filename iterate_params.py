import subprocess as sp
import numpy
import shlex
from time import sleep
import os
import os.path as path
import re
import pdb
import esq_construct_ukln as construct_ukln


def C6_C12(epsi,sig):
    C6 = 4*epsi*sig**6
    C12 = 4*epsi*sig**12
    return C6, C12
def rsync(files, dest='.', flags=None, direction='to'):
    #Make an rsync move to or from fir
    #files is a string that is the exact command that you send to fluorine
    basersync = 'rsync -e \'"ssh -i /home/ln8dc/.ssh/fluorinekey"\' ' #required space
    #The escaped single quote around the double is required to get the shlex processor to actually preserve the double quote in the argment pass... a bit annoying, but it works. Without it, the "..." is correctly parsed as a single token, but then the token is passed as a plain string without the "" causing the --rsh command to only interpret the "ssh" part.
    basersync = 'rsync "-e ssh -i /home/ln8dc/.ssh/fluorinekey" ' #required space
    if flags is not None:
        basersync += flags
    if direction is 'to':
        cmd = basersync + files + ' fir.itc.virginia.edu:/home/ln8dc/ljspherespace/{0}'.format(dest)
    else:
        cmd = basersenc + ' fir.itc.virginia.edu:/home/ln8dc/ljspherespace/{0}'.format(dest) + files + ' .'
    args = shlex.split(cmd)
    p = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE, shell=False).wait()
    #(stdout, stderr) = p.communicate()
def runsh(cmdstr, IO=False, delay=0.2):
    #Run an command based on the cmd string
    args = shlex.split(cmdstr)
    p = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE)
    std = os.waitpid(p.pid, 0)
    if IO is True:
        (stdout, stderr) = p.communicate()
        #p.terminate()
        return stdout, stderr
    else:
        #p.terminate()
        return
def gen_file(filepath, Atype, charge, skel):
    top = open(filepath, 'w')
    #Update LJX
    output = skel.replace('LJS_LJX', Atype)
    #Update Charge
    output = re.sub('SY.Z+', '{0:+.4f}'.format(charge), output)
    top.write(output)
    top.close()
    return

def iterate(continuation=True, start=None):
    basessh = 'ssh -i /home/ln8dc/.ssh/fluorinekey fir.itc.virginia.edu ' #Space is required, dont forget it
    #1) Start with initial parameters (determined pre-simulations)
    #q,e,s
    if not continuation:
        parms = numpy.load('n21_init.npy')
    else:
        parms = numpy.load('qes.npy')
    qs = parms[:,0]
    es = parms[:,1]
    ss = parms[:,2]
    nlj = len(qs)
    C6, C12 = C6_C12(es,ss)
    #2) Run the Equilibration of new states
    #    2a) Wait
    #3) Run the production of the new states
    #    3a) Wait on NEW states
    #4) Old States: Run the kln through the new states
    #5) After new prod is done...
    #6) Find g_t for all new states
    #7) Subsample new states with the g_t
    #8) Rerun new states
    #    8a) Run new states through kln ALL states (old and new)
    #    8b) Run new states through rep
    #    8c) Run new states through null
    #    8d) Run new states through q/q2
    #9) Copy xvg files from ALL states to Fl
    #    9a) Copy all files to argon
    #10) Update esq_construct_ukln.py with the parameters from the new states
    #    10a) Update nstates
    #    10b) Use excel sheet as needed
    if start is None or start == 'free energy':
        pdb.set_trace()
        #11) Run esq_construct_ukln.py to determine basis functions, free energies
        #    11a) Generate the movie
        #    11c) Find the resample points
        #Run the main crunch script
        construct_ukln(nlj, qs, es, ss)
        newresamp = numpy.load('resamp_points_n{0:d}'.format(nlj))
        qs = numpy.append(qs, newresamp[:,0])
        es = numpy.append(es, newresamp[:,1])
        ss = numpy.append(ss, newresamp[:,0])
        nlj = len(qs)
        newparm=numpy.zeros([nlj,3])
        newparm[:,0] = qs
        newparm[:,1] = es
        newparm[:,2] = ss
        numpy.save('qes.npy', newparm)
        numpy.savetxt('qes.txt', newparm)
        #12) Generate new C6/C12
        C6, C12 = C6_C12(es,ss)
    #14) Update the new states on the skeleton generator/copier
    if start is None or start == 'topology':
        #~~Generate Topologies~~
        skeltop = open('ljspheres_esq_skel.top').read()
        replacestr = ''
        #Fill in the atomtype entries
        for i in xrange(nlj):
            replacestr+='LJS_LJ{0:d}     LJS      0.0000  0.0000  A   {1:.5E}  {2:.5E};\n'.format(i, C6[i], C12[i])
            if i == 0 or i == 5:
                replacestr+='rep_LJ{0:d}     LJS      0.0000  0.0000  A   {1:.5E}  {2:.5E};\n'.format(i, 0, C12[i])
        skel = skeltop.replace('REPLACE', replacestr)
        for i in xrange(nlj): #Create the files
            folder = '/home/ln8dc/simulations/ion-parameters/lj{0:d}'.format(i)
            Atype = 'LJS_LJ{0:d}'.format(i)
            try: #Make the folder if needed
                os.mkdir(folder)
            except:
                pass
            filename = 'ljspheres_esq.top'
            filepath = path.join(folder, filename)
            #Copy the old file
            gen_file(filepath, Atype, qs[i], skel)
            #Make the correct null topology file
            filename = 'ljspheres_null_esq.top'
            filepath = path.join(folder, filename)
            gen_file(filepath, 'null_LJ0', 0, skel)
            #make the correct nullq topology file
            filename = 'ljspheres_nullq_esq.top'
            filepath = path.join(folder, filename)
            gen_file(filepath, 'null_LJ0', 1.0000, skel)
            #make the correct nullq2 topology file
            filename = 'ljspheres_nullq2_esq.top'
            filepath = path.join(folder, filename)
            gen_file(filepath, 'null_LJ0', 0.5000, skel)
            #Create special repulsive only cases
            if i == 5 or i == 0 :
                filename = 'ljspheres_rep_esq.top'
                filepath = path.join(folder,filename)
                gen_file(filepath, 'rep_LJ{0:d}'.format(i), 0, skel)
        #15) Run the skeleton generator and copier
            topoflags='--include="lj*" --include="*.top" --exclude="*" '
            rsync('/home/ln8dc/simulations/ion-parameters/lj{0:d}'.format(i), direction='to', flags='-rv '+topoflags) #Creates folder and sends topologies.
        pdb.set_trace()
        runsh(basessh + 'sh /home/ln8dc/ljspherespace/copy_skel.sh {0:d}'.format(nlj-1)) #Generates the skelton script
    #16) Repeat from Step 2


    #topname = 'ljspheres_esq_noid.top'
    #topfile = open(topname,'w')
    #topfile.write(skeltop.replace('REPLACE',replacestr))
    #topfile.close(
    ##send topology file
    #scp(topname,direction='to')
    ##Generate remote skeleton structure
    #cmd = basessh + 'python /home/ln8dc/ljspherespace/gen_esq_id.py'
    

if __name__ == "__main__":
    continuation = False
    iterations = 1
    startfrom = 'topology'
    for i in xrange(iterations):
        iterate(continuation=continuation, start=startfrom)
        continuation = True #REQUIRED, ensures init parms are never processed more than once
        startfrom = None
