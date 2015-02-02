import subprocess as sp
import numpy
import shlex
from time import sleep
import os
import os.path as path
import re
import pdb
import esq_construct_ukln as construct_ukln

basessh = 'ssh -i /home/ln8dc/.ssh/fluorinekey fir.itc.virginia.edu ' #Space is required, dont forget it

def C6_C12(epsi,sig):
    C6 = 4*epsi*sig**6
    C12 = 4*epsi*sig**12
    return C6, C12
def rsync(files, dest='.', flags=None, direction='to'):
    #Make an rsync move to or from fir
    #files is a string that is the exact command that you send to fluorine
    #basersync = 'rsync -e \'"ssh -i /home/ln8dc/.ssh/fluorinekey"\' ' #required space
    #The escaped single quote around the double is required to get the shlex processor to actually preserve the double quote in the argment pass... a bit annoying, but it works. Without it, the "..." is correctly parsed as a single token, but then the token is passed as a plain string without the "" causing the --rsh command to only interpret the "ssh" part.
    basersync = 'rsync "-e ssh -i /home/ln8dc/.ssh/fluorinekey" ' #required space
    if flags is not None:
        basersync += flags
    if direction is 'to':
        cmd = basersync + files + ' fir.itc.virginia.edu:/home/ln8dc/ljspherespace/{0}'.format(dest)
    else:
        cmd = basersenc + ' fir.itc.virginia.edu:/home/ln8dc/ljspherespace/{0}'.format(dest) + files + ' .'
    args = shlex.split(cmd)
    p = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE, shell=False).communicate()
    #(stdout, stderr) = p.communicate()
    return
def runsh(cmdstr, IO=False, delay=0.2, stdout=sp.PIPE, stderr=sp.PIPE):
    #Run an command based on the cmd string
    args = shlex.split(cmdstr)
    #std = os.waitpid(p.pid, 0)
    if IO is True:
        p = sp.Popen(args, stdout=stdout, stderr=stderr)
        (stdout, stderr) = p.communicate()
        #p.terminate()
        return stdout, stderr
    else:
        p = sp.Popen(args, stdout=stdout, stderr=stderr).communicate()
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
def jobids(qstat_str):
    return [re.sub(r'(^\d+.lc5).*',r'\1',line) for line in qstat_str.split('\n') if re.match(r'^\d+\.lc5',line) is not None]
    
def monitor_jobs(jobs, delay=60):
    #Querey the clutser's job id's until all jobs complete
    #Accepts the list of job id's
    all_done = False
    while not all_done:
        (jobout,errout) = runsh(basessh + 'qstat -u ln8dc', IO=True)
        #Process each line, grabbing only the lc5 from the lines
        alllines = jobids(jobout)
        jobsdone= []
        for job in jobs:
            if not job in alllines:
                jobsdone.append(job)
        for job in jobsdone:
            jobs.remove(job)
        if len(jobs) == 0:
            all_done=True
            break
        else:
            #print "The following jobs are still running:"
            #print jobs
            sleep(delay)
    return
            

def iterate(continuation=True, start=None):
    #1) Start with initial parameters (determined pre-simulations)
    #q,e,s
    if not continuation:
        parms = numpy.load('n21_init.npy')
        old_nlj = 0
    else:
        parms = numpy.load('qes.npy')
        (old_nlj, newnlj) = numpy.load('old_new_nlj.npy')
    qs = parms[:,0]
    es = parms[:,1]
    ss = parms[:,2]
    nlj = len(qs)
    C6, C12 = C6_C12(es,ss)
    #Initilization which will start 
    if start is None or start == 'equilibrate':
        print "Equilibrating"
        runsh(basessh + 'sh /home/ln8dc/ljspherespace/copy_skel.sh {0:d}'.format(nlj-1)) #Generates the skelton script
        #2) Run the Equilibration of new states
        #    2a) Wait
        #Start the execution of jobs
        (out, err) = runsh(basessh + 'sh /home/ln8dc/ljspherespace/submit_equil_esq.sh {0:d} {1:d}'.format(old_nlj, nlj-1), IO=True) #Generates the skelton script
        lstout = [re.sub(r'(^\d+.lc5).*',r'\1',line) for line in out.split('\n') if re.match(r'^\d+\.lc5',line) is not None]
        #Add a delay to make sure the jobs make it to the que (~5-10 seconds seems to be enough)
        sleep(8)
        #Monitor for job completion
        monitor_jobs(lstout)
        #Cleanup
        runsh(basessh + 'rm /bigtmp/ln8dc/ljsphere_es/equilibrate*')
        start = None
    if start is None or start == 'production':
        print "Production Simulation"
        #3) Run the production of the new states
        #    3a) Wait on NEW states
        #Start the execution of production
        (out, err) = runsh(basessh + 'sh /home/ln8dc/ljspherespace/submit_lj_esq.sh {0:d} {1:d}'.format(old_nlj, nlj-1), IO=True) #Generates the skelton script
        lstout = jobids(out)
        #Add a delay to make sure the jobs make it to the que (~5-10 seconds seems to be enough)
        sleep(8)
        #Monitor for job completion
        monitor_jobs(lstout)
        start = None
    if start is None or start =='g_t':
        print "Measuring Timeseries"
        #6) Find g_t for all new states
        runsh(basessh + '\"cd /home/ln8dc/ljspherespace; python /home/ln8dc/ljspherespace/find_g_t.py {0:d} {1:d}\"'.format(old_nlj,nlj))
        start = None
    if start is None or start == 'subsample':
        print "Subsampling"
        #7) Subsample new states with the g_t
        (subout, suberr) = runsh(basessh + '\"cd /home/ln8dc/ljspherespace; sh /home/ln8dc/ljspherespace/submit_subsample.sh {0:d} {1:d}\"'.format(old_nlj,nlj-1), IO=True)
        sleep(8)
        lstout = jobids(subout)
        monitor_jobs(lstout)
        start = None
    if start is None or start == 'kln':
        print "Rerunning"
        #4) Old States: Run the kln through the new states
        basekln = '\"cd /home/ln8dc/ljspherespace; sh /home/ln8dc/ljspherespace/submit_optrerun_esq.sh FLAGS\"'
        flagstr = '-lk {lk:d} -uk {uk:d} -ll {ll:d} -ul {ul:d} {tpr:s} {submit:s} {kln:s} {null:s} {qq2:s} {rep:s}'
        if not continuation:
            oldflags = {'lk':0, 'uk':nlj-1, 'll':old_nlj, 'ul':nlj-1, 'tpr':'--tpr', 'submit':'--submit', 'kln':'--kln', 'null':'--null', 'qq2':'--qq2', 'rep':'--rep'}
        else:
            oldflags = {'lk':0, 'uk':old_nlj-1, 'll':old_nlj, 'ul':nlj-1, 'tpr':'--tpr', 'submit':'--submit', 'kln':'--kln', 'null':'', 'qq2':'', 'rep':''}
        (oldout, olderr) = runsh(basessh + basekln.replace('FLAGS',flagstr.format(**oldflags)), IO=True)
        if continuation:
            #8) Rerun new states
            #    8a) Run new states through kln ALL states (old and new)
            #    8b) Run new states through rep
            #    8c) Run new states through null
            #    8d) Run new states through q/q2
            newflags = {'lk':old_nlj, 'uk':nlj-1, 'll':0, 'ul':nlj-1, 'tpr':'--tpr', 'submit':'--submit', 'kln':'--kln', 'null':'--null', 'qq2':'--qq2', 'rep':'--rep'}
            (newout, newerr) = runsh(basessh + basekln.replace('FLAGS',flagstr.format(**newflags)), IO=True)
        else:
            newout = ''
            newerr = ''
        totalout = oldout + '\n' + newout
        klnids = jobids(totalout)
        sleep(8)
        monitor_jobs(klnids)
        #Cleanup
        runsh(basessh + 'rm /bigtmp/ln8dc/ljsphere_es/rerun*')
        start = None
    if start is None or start == 'xvgcopy':
        print "Moving XVG Files"
        #9) Copy xvg files from ALL states to Fl
        #    9a) Copy all files to argon
        xvgflags='--exclude="*.top" --include="lj*" --include="prod" --include="*.xvg" --exclude="*" '
        basersync = 'rsync "-e ssh -i /home/ln8dc/.ssh/fluorinekey" -rv ' #required space
        cmd = basersync + xvgflags + ' fir.itc.virginia.edu:/bigtmp/ln8dc/ljsphere_es/.' + ' .' #Grab xvg files from bigtmp and move them here
        args = shlex.split(cmd)
        p = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE, shell=False).communicate()
        start=None
    #10) Update esq_construct_ukln.py with the parameters from the new states
    #    10a) Update nstates
    #    10b) Use excel sheet as needed
    if start is None or start == 'free energy':
        print "Computing Free Energy"
        #11) Run esq_construct_ukln.py to determine basis functions, free energies
        #    11a) Generate the movie
        #    11c) Find the resample points
        #Run the main crunch script
        construct_ukln.execute(nlj, qs, es, ss)
        newresamp = numpy.load('resamp_points_n{0:d}.npy'.format(nlj))
        qs = numpy.append(qs, newresamp[:,0])
        es = numpy.append(es, newresamp[:,1])
        ss = numpy.append(ss, newresamp[:,2])
        new_nlj = len(qs)
        numpy.save('old_new_nlj.npy', numpy.array([nlj, new_nlj]))
        nlj=new_nlj
        newparm=numpy.zeros([nlj,3])
        newparm[:,0] = qs
        newparm[:,1] = es
        newparm[:,2] = ss
        numpy.save('qes.npy', newparm)
        numpy.savetxt('qes.txt', newparm)
        #12) Generate new C6/C12
        C6, C12 = C6_C12(es,ss)
        start = None
    #14) Update the new states on the skeleton generator/copier
    if start is None or start == 'topology':
        print "Generating Topologies"
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
    continuation = True
    iterations = 4
    startfrom = 'free energy'
    for i in xrange(iterations):
        iterate(continuation=continuation, start=startfrom)
        continuation = True #REQUIRED, ensures init parms are never processed more than once
        startfrom = None
