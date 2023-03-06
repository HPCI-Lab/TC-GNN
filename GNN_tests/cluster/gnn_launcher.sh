#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --error=job.err            # standard error file
#SBATCH --output=job.out           # standard output file

module load torch/1.9.0a0

#python patch_fetch.py
#python csv_test.py
python ./src/main.py
