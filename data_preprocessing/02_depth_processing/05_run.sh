#!/bin/bash
#SBATCH -A cli138
#SBATCH -J 05_run
#SBATCH -N 1
#SBATCH -t 8:00:00
#SBATCH -o ./slurm_output/05_run-output.txt
#SBATCH -e ./slurm_output/05_run-error.txt
    
# module load cuda/11.0.2
conda init bash
source ~/.bashrc
conda activate triton

python -u 05_extract_raster_from_netcdf.py

