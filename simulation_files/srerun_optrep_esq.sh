#!/bin/sh
#SBATCH -p parallel
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
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

#Rerun all systems with the REPUSLIVE congif
for k in $(seq LK UK)
do
    cd $SLURM_SUBMIT_DIR 
    dir=lj"$k"
    cd $dir/prod
    for l in 5 0
    do
        mdrun_d -v -pin off -ntmpi 8 -rerun subprod"$k".trr -deffnm subprod"$k"_"$l"_rep
        echo 5 0 |g_energy_d -dp -f subprod"$k"_"$l"_rep.edr -o subenergy"$k"_"$l"_rep.xvg
    done
done
