#!/bin/sh
#SBATCH -p serial
#SBATCH --time=0-4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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

#Run through all systems and convert to the subsampled form

startI=XX
endI=YY

for k in $(seq $startI $endI)
do
    skipnum=$(head -n1 /scratch/ln8dc/ljsphere_es/lj$k/prod/g_t$k.txt)
    cd "/scratch/ln8dc/ljsphere_es/lj$k/prod"
    trjconv_d -skip "$skipnum" -f "prod$k.trr" -o "subprod$k.trr"
done
