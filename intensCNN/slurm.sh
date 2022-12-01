#!/bin/bash -l

#SBATCH --partition=p.ada
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mail-type=none

#SBATCH --time=6:0:0
#SBATCH -J simDecoder
#SBATCH -o .%j.out
#SBATCH -e .%j.out

module purge
module load anaconda/3/2021.11 cuda/11.6
source activate /u/kayyer/conda-envs/cnicuda

python train.py
