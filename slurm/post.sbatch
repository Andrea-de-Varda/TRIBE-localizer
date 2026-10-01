#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH -t 03:00:00
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --job-name=tribeloc_post
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Univariate analysis (main): task maps for all items and each item half, domain contrasts with FWE,
# parcel summaries (whole parcel + held-out fROI), spin tests, figures. CPU only, needs only the shards.
#   sbatch slurm/post.sbatch
source slurm/common.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python scripts/06_univariate.py
python scripts/08_univariate_summary.py
python scripts/09_figures.py
