#!/bin/sh

#Convert the X to the correct 

if [ -z "$1" ] ; then
    startI=36
else
    startI=$1
fi
if [ -z "$2"] ; then
    endI=45
else
    endI=$2
fi

#for i in {36..45}
for i in $(seq $startI $endI)
do
    #Create directory
    cd /scratch/ln8dc/ljsphere_es
    dir=lj"$i"
    mkdir -p $dir/prod
    #Copy over mdp file
    cp /home/ln8dc/ljspherespace/md_es.mdp $dir/prod/md_es.mdp
    #Copy pbs script to location
    cat /home/ln8dc/ljspherespace/srun_ljX_es.sh | sed "s:XX:$i:g" > $dir/prod/run_lj"$i".sh
    cd $dir/prod
    grompp_d -f md_es.mdp -p ../ljspheres_esq.top -c ../npteq/npt"$i".gro -o prod"$i".tpr -maxwarn 1
    #Submit job
    jobid=$(sbatch run_lj"$i".sh)
    echo "$jobid"
    #echo "$jobid" | sed "s:lc5.*:lc5:" >> runlist.txt
    #touch $jobid
done
