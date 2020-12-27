#!/bin/bash
#SBATCH -A research
#SBATCH --job-name=___
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --output=output_logs_files/glow__e_10_%j.txt       # Output file.
#SBATCH --mail-type=END                # Enable email about job finish 
#SBATCH --mail-user=sandeep.nagar@research.iiit.ac.in    # Where to send mail  
#SBATCH --mem-per-cpu=3000
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cudnn/7-cuda-10.0

mpiexec -n 1 python glow.py --train --evaluate --generate #--visualize
