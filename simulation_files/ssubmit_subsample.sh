#!/bin/sh

if [ -z "$1" ] ; then
    startI=0
else
    startI=$1
fi
if [ -z "$2" ] ; then
    endI=0
else
    endI=$2
fi

#Copy subsample skel

cat /home/ln8dc/ljspherespace/ssubsampleXY_skel.sh | sed "s:XX:$startI:" | sed "s:YY:$endI:" > /home/ln8dc/ljspherespace/subsampleRun.sh
jobid=$(sbatch /home/ln8dc/ljspherespace/subsampleRun.sh)
echo $jobid
#Cleanup
rm /home/ln8dc/ljspherespace/subsampleRun.*

