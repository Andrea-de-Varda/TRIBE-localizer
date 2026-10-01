#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH --gres=gpu:a100:1
#SBATCH -t 02:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --job-name=tribeloc_baseline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# One GPU: no-input baseline (12), then on the same node the whole CPU chain with baseline subtraction:
# task means (06), contrast maps (07), localizer fROIs (11), parcel summaries (08), figures (09).
#   sbatch slurm/baseline.sbatch
source slurm/common.sh
python scripts/12_baseline.py
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python scripts/06_univariate.py
python scripts/07_univariate_contrasts.py
python scripts/11_localizers.py
python scripts/08_univariate_summary.py
python scripts/09_figures.py
