#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH -t 06:00:00
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --job-name=tribeloc_post
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# After all searchlight tasks: domain maps, univariate maps, parcel summaries and spin tests.
#   sbatch --dependency=afterok:<searchlight array job id> slurm/post.sbatch
source slurm/common.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python scripts/05_domain_maps.py
python scripts/06_univariate.py
python scripts/07_parcel_summary.py
