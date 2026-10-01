#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH -t 02:00:00
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH --job-name=tribeloc_qc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Quality check of all TRIBE shards -> results/analysis/qc/ (CPU only).
source slurm/common.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/03b_qc_shards.py
