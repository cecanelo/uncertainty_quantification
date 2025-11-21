#!/bin/bash
#SBATCH --job-name=canelo_uq_job
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=canelo@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# Activate your conda environment
source activate thesis

# Run your python script
srun python my_main_script.py
