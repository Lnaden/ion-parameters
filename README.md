ion-parameters
==============

This houses the main scripts I use for creating the iterative ion parametrization.

Primary script is iterate\_params.py (for PBS job handling on UVa Fir Custer) or siterate\_params.py (for SLURM job handling on UVa Rivanna Cluster)

Script is designed to configure and execute jobs on the clusters from remote computer with SSH Key access.

Analysis documentation is contained within the esq\_construct\_ukln.py script. Please refer to it for how to call and use the documentation.

Individual GROMACS run have not been included in this repo as they are specific to the version of GROMACS used to generate the data and not needed the primary focus of the methods in the analysis. 
GROMACS files can be provided upon request.
