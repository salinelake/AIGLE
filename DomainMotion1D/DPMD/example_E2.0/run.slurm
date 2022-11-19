#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 25:0:0
##SBATCH --mem 128G 
#SBATCH --gres=gpu:1
#SBATCH --job-name=dmotion

module purge
module load anaconda3
conda activate dp-dev


unlink frozen_model.pb
ln -s ../DP.pb frozen_model.pb

seed=$RANDOM
echo $seed > seed.txt
mpirun -n 1 lmp -v TEMP 300 -v SEED $seed -in in.lammps > pto.log &
 
wait
