#!/bin/sh
#SBATCH -p parallel
#SBATCH --time=0-20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=levinaden+simulation.reporter@gmail.com
#SBATCH --no-requeue

# location of GROMACS binaries on cluster: /h3/n1/shirtsgroup/gromacs_46/install/bin/
# location of GROMACS top files on cluster: /h3/n1/shirtsgroup/gromacs_46/install/share/gromacs/top

#module load mpich3-gnu
#cat $$PBS_NODEFILE

module load openmpi/gcc/1.8.1
export OMP_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR

mdrun_d -v -pin off -ntmpi 16 -deffnm prodXX
echo 5 0 | g_energy_d -dp -f prodXX.edr -o base_energyXX.xvg
