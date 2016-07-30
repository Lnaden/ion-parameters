#!/bin/sh

#Sections:
## Rerun A-B through X-Y
## Rerun A-B through Null
## Rerun A-B through q and q2
## Rerun A-B through Rep 0 and Rep 10

dokln=false
donull=false
doqq2=false
dorep=false
createtpr=false
submitjobs=false

while test $# -gt 0; do
    case "$1" in
        -lk|--lowerk)
            shift
            lowerk=$1
            shift
        ;;
        -uk|--uperk)
            shift
            upperk=$1
            shift
        ;;
        -ll|--lowerl)
            shift
            lowerl=$1
            shift
        ;;
        -ul|--uperl)
            shift
            upperl=$1
            shift
        ;;
        --kln)
            shift
            dokln=true
        ;;
        --null)
            shift
            donull=true
        ;;
        --qq2)
            shift
            doqq2=true
        ;;
        --rep)
            shift
            dorep=true
        ;;
        --submit)
            shift
            submitjobs=true
        ;;
        --tpr)
            shift
            createtpr=true
        ;;
        *)
            break
        ;;
    esac
done



#Flags to do the various parts

if $createtpr
then
    ##Rerun A-B through X-Y
    if $dokln
    then
        for k in $(seq $lowerk $upperk) #A-B
        do
            cd /scratch/ln8dc/ljsphere_es
            dir=lj"$k"
            #Copy over mdp file supressing cord output
            cp /home/ln8dc/ljspherespace/md_nocord_es.mdp $dir/prod/md_nocord_es.mdp
            cd $dir/prod
            for l in $(seq $lowerl $upperl) #X-Y
            do
                #Run setup the the deltaF to run the uncharged through charged
                grompp_d -f md_nocord_es.mdp -p ../../lj"$l"/ljspheres_esq.top -c prod"$k".gro -o subprod"$k"_"$l" -maxwarn 1
                #Allow me to bypass the 99 backups of mdout rule
                rm \#mdout*
            done
            #Clean up #mdout
            rm \#*
        done
    fi
    ##################################################################################################################################
    ##Rerun A-B through Null
    if $donull
    then
        for k in $(seq $lowerk $upperk) #A-B
        do
            cd /scratch/ln8dc/ljsphere_es
            dir=lj"$k"
            cp /home/ln8dc/ljspherespace/md_nocord_es.mdp $dir/prod/md_nocord_es.mdp
            cd $dir/prod
            grompp_d -f md_nocord_es.mdp -p ../ljspheres_null_esq.top -c prod"$k".gro -o subprod"$k"_null.tpr -maxwarn 1
            #Clean up #mdout
            rm \#*
        done
    fi
    ##################################################################################################################################
    #Rerun A-B through q and q2
    if $doqq2
    then
        for k in $(seq $lowerk $upperk) #A-B
        do
            cd /scratch/ln8dc/ljsphere_es
            dir=lj"$k"
            #Copy over mdp file supressing cord output
            cp /home/ln8dc/ljspherespace/md_nocord_es.mdp $dir/prod/md_nocord_es.mdp
            cd $dir/prod
            grompp_d -f md_nocord_es.mdp -p ../ljspheres_nullq_esq.top -c prod"$k".gro -o subprod"$k"_q -maxwarn 1
            grompp_d -f md_nocord_es.mdp -p ../ljspheres_nullq2_esq.top -c prod"$k".gro -o subprod"$k"_q2 -maxwarn 1
            #Clean up #mdout
            rm \#*
        done
    fi
    ##################################################################################################################################
    ## Rerun A-B through Rep 0 and Rep 10
    if $dorep
    then
        for k in $(seq $lowerk $upperk) #A-B
        do
            cd /scratch/ln8dc/ljsphere_es
            dir=lj"$k"
            cd $dir/prod
            for l in 0 5
            do
                grompp_d -f md_nocord_es.mdp -p ../../lj"$l"/ljspheres_rep_esq.top -c prod"$k".gro -o subprod"$k"_"$l"_rep.tpr -maxwarn 1
            done
            #Cleanup #mdout
            rm \#*
        done
    fi
fi
##################################################################################################################################
##################################################################################################################################


#Submit the 4 jobs
cd /scratch/ln8dc/ljsphere_es

if $submitjobs
then
    ##Rerun A-B through X-Y
    if $dokln
    then
        cat /home/ln8dc/ljspherespace/srerun_optkln_esq.sh | sed "s:LK:$lowerk:" |sed "s:UK:$upperk:" |sed "s:LL:$lowerl:"| sed "s:UL:$upperl:" > rerun_kln_esq.sh
        jobid=$(sbatch rerun_kln_esq.sh)
        echo "$jobid"
        #touch kln_$jobid
    fi
    
    ##Rerun A-B through Null
    if $donull
    then
        cat /home/ln8dc/ljspherespace/srerun_optnull_esq.sh | sed "s:LK:$lowerk:" |sed "s:UK:$upperk:" > rerun_null_esq.sh
        jobid=$(sbatch rerun_null_esq.sh)
        echo "$jobid"
        #touch null_$jobid
    fi
    
    ##Rerun A-B through q and q2
    if $doqq2
    then
        cat /home/ln8dc/ljspherespace/srerun_optq_esq.sh | sed "s:LK:$lowerk:" |sed "s:UK:$upperk:" > rerun_q_esq.sh
        jobid=$(sbatch rerun_q_esq.sh)
        echo "$jobid"
        #touch null_$jobid
    fi
    
    ## Rerun A-B through Rep 0 and Rep 10
    if $dorep
    then
        cat /home/ln8dc/ljspherespace/srerun_optrep_esq.sh | sed "s:LK:$lowerk:" |sed "s:UK:$upperk:" > rerun_rep_esq.sh
        jobid=$(sbatch rerun_rep_esq.sh)
        echo "$jobid"
        #touch rep_$jobid
    fi
fi
