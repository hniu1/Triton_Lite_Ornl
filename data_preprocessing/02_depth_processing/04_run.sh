#!/bin/bash
#SBATCH -A cli138
#SBATCH -J 04_run
#SBATCH -N 1
#SBATCH -t 8:00:00
#SBATCH -o ./slurm_output/04_run-output.txt
#SBATCH -e ./slurm_output/04_run-error.txt
    
# module load cuda/11.0.2
conda init bash
source ~/.bashrc
conda activate triton

python -u 04_block_selection_to_csv_export.py

