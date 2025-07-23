#!/bin/bash

#SBATCH -A inai
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --partition=ihub
#SBATCH --output=dense_reconst.txt
#SBATCH --nodelist=gnode098
#SBATCH --job-name=dense_reconst

echo "Running on $(hostname)"
echo "Job started at $(date)"

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

cd /scratch/Ananya_Kulkarni_AWR/MAP_LITE_IND/

python dense_reconstruction.py 

echo "Dense reconstruction completed at $(date) and temporary files cleaned up."
