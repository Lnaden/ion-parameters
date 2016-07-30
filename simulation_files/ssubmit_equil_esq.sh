#!/bin/sh


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

#Run through the LJ of interest
#for i in {36..45}
for i in $(seq $startI $endI)
do
    cd /home/ln8dc/ljspherespace/
    dir=lj"$i"
    cat sequilibrate_esqXX.sh | sed "s:XX:$i:g" > /scratch/ln8dc/ljsphere_es/"equilibrate_esq$i.sh"
    cd /scratch/ln8dc/ljsphere_es
    jobid=$(sbatch "equilibrate_esq$i.sh")
    #echo "$jobid" - started on `date` | sed "s:lc5.*:lc5:" >> eqlist.txt
    echo "$jobid"
    #Cleanup
    rm "equilibrate_esq$i.sh"
done
