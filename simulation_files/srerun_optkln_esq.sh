#!/bin/sh
#SBATCH -p parallel
#SBATCH --time=0-40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=ln8dc@virginia.edu
#SBATCH --no-requeue

# location of GROMACS binaries on cluster: /h3/n1/shirtsgroup/gromacs_46/install/bin/
# location of GROMACS top files on cluster: /h3/n1/shirtsgroup/gromacs_46/install/share/gromacs/top

#module load mpich3-gnu
#cat $$PBS_NODEFILE

module load openmpi/gcc/1.8.1
export OMP_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR

#Rerun all systems with the charge
for k in $(seq LK UK)
do
    cd $SLURM_SUBMIT_DIR 
    dir=lj"$k"
    cd $dir/prod
    for l in $(seq LL UL)
    do
        mdrun_d -v -pin off -ntmpi 16 -rerun subprod"$k".trr -deffnm subprod"$k"_"$l"
        echo 5 0 |g_energy_d -dp -f subprod"$k"_"$l".edr -o subenergy"$k"_"$l".xvg
    done
done
