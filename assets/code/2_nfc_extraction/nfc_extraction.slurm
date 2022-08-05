#!/usr/bin/env bash
#SBATCH --job-name=extraction
#SBATCH --output=nfc_extraction.out
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --cluster=smp
#SBATCH --partition=smp

module purge
module load python/3.7.0 venv/wrap

workon opso_dev

python nfc_extraction.py

crc-job-stats.py
