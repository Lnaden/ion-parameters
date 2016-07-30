#!/bin/sh
#SBATCH -p parallel
#SBATCH --time=0-6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=ln8dc@virginia.edu
#SBATCH --no-requeue

# location of GROMACS binaries on cluster: /h3/n1/shirtsgroup/gromacs_46/install/bin/
# location of GROMACS top files on cluster: /h3/n1/shirtsgroup/gromacs_46/install/share/gromacs/top

module load openmpi/gcc/1.8.1
export OMP_NUM_THREADS=1
#cat $$PBS_NODEFILE

cd $SLURM_SUBMIT_DIR

dir=ljXX
mkdir -p $dir/nvteq $dir/npteq
cp /home/ln8dc/ljspherespace/nvt_es.mdp $dir/nvteq/nvt_es.mdp
cp /home/ln8dc/ljspherespace/npt_es.mdp $dir/npteq/npt_es.mdp

cd $dir/nvteq #NPT Eq
grompp_d -f nvt_es.mdp -p ../ljspheres_esq.top -c /scratch/ln8dc/ljsphere_es/npt1.gro -o nvtXX.tpr
mdrun_d -v -pin on -ntmpi 16 -deffnm nvtXX

cd ../npteq #NPT Eq
grompp_d -f npt_es.mdp -p ../ljspheres_esq.top -c ../nvteq/nvtXX.gro -o nptXX.tpr
mdrun_d -v -pin on -ntmpi 16 -deffnm nptXX
