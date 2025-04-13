#!/bin/bash
#SBATCH --job-name=algorithm1
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=24:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=kd5846@princeton.edu

module load python/3.9
source ~/myenv/bin/activate

srun --ntasks=64 --exclusive bash -c 'python memorization_chunk.py $SLURM_PROCID 64' &
wait

