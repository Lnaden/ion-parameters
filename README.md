ion-parameters
==============

This houses the main scripts I use for creating the iterative ion parametrization.

Primary script is iterate\_params.py (for PBS job handling on UVa Fir Custer) or siterate\_params.py (for SLURM job handling on UVa Rivanna Cluster)

Script is designed to configure and execute jobs on the clusters from remote computer with SSH Key access.

Analysis documentation is contained within the esq\_construct\_ukln.py script. Please refer to it for how to call and use the documentation.
A sereies of example data has been provided in the example\_init\_data/ folder which will run with the esq\_construct\_ukln.py script to show how the script funcitons. Please see the readme in the folder for instructions on how to use the sample data.

Individual GROMACS run have been included in this repo in the simulation\_files directory, but are specific to the version of GROMACS used to generate the data and will probably not work as is.
