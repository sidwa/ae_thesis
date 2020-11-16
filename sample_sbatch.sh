#!/bin/bash -l
# NOTE the -l flag!


# This is an example job file for a single core CPU bound program
# Note that all of the following statements below that begin
# with #SBATCH are actually commands to the SLURM scheduler.
# Please copy this file to your home directory and modify it
# to suit your needs.
 
# If you need any help, please email rc-help@rit.edu
# Name of the job - You'll probably want to customize this.
##SBATCH --job-name=data_parallel_test

 # Standard out and Standard Error output files
##SBATCH --output=data_parallel_test.out
##SBATCH --error=data_parallel_test.err

#To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user=sxr8618@rit.edu
 
# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL
 
# 5 days is the run time MAX, anything over will be KILLED unless you talk RC
# Request Time limit day-hrs:min:sec
#SBATCH --time=0-1:0:0

# Put the job in the partition that matches the account and request one core
#SBATCH --account=ganvoice 
#SBATCH -p debug
#SBATCH --cpus-per-task=1 #Advise the Slurm controller that ensuing job steps will require ncpus number of processors per task.  Default is 1 processor per task.

# Job memory requirements in MB <default>, MB=m, GB=g, or TB=t
#SBATCH --mem=16g
#SBATCH --gres=gpu:p4:1

spack env activate ganvoice

 #
# Your job script goes below this line.  
#

python cifar10.py --batch_size $1 --num_epochs $2