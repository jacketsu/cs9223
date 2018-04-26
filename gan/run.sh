#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=GPUDemo
#SBATCH --mail-type=END
#SBATCH --mail-user="email address"
#SBATCH --output=slurm_%j.out

cd "path-to-program"
source activate ML
python train_estimator.py
