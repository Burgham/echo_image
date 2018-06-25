#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=echo
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=6GB 

source activate py36

python vgg16_echo.py
