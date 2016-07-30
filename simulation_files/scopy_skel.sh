#!/bin/sh

#Accepts single digit command line argument to copy over skeleton structure to bigtimp
#Goes from 0 -> ARG
for i in $(seq 0 $1)
do
    cp -r /home/ln8dc/ljspherespace/lj"$i" /scratch/ln8dc/ljsphere_es
done
