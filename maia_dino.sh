#!/bin/bash
# The interpreter used to execute the script
#\#SBATCH" directives that convey submission options:
#SBATCH --job-name=maia_dino
#SBATCH --mail-user=yuic@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4000m
#SBATCH --time=01:00:00
#SBATCH --account=cse598f25s014_class
#SBATCH --partition=gpu_mig40,spgpu
#SBATCH --gpus=1
#SBATCH --output=./log/dino/maia_dino.log

echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Time is: $(date)"
echo "Directory is: $(pwd)"


# Activate your environment
source ~/.bashrc
conda activate maia

python main.py --model dino_vits8 --unit_mode manual --path2save ./results/dino --units blocks.7.mlp.fc1=77 > ./log/dino/maia_dino.txt

echo "Job finished with exit code $? at: $(date)"