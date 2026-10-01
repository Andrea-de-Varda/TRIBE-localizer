#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH --gres=gpu:a100:1
#SBATCH -t 03:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --job-name=tribeloc_localizers
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# One GPU: TRIBE predictions for the localizer stimuli (4 shards, ~1,100 stimuli), then on the same node the
# CPU steps: localizer contrasts and fROIs (11), parcel summaries with fROIs (08), figures (09).
#   sbatch slurm/localizers.sbatch
source slurm/common.sh
python scripts/03_run_tribe.py --localizers --shards 0 1 2 3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python scripts/11_localizers.py
python scripts/08_univariate_summary.py
python scripts/09_figures.py
