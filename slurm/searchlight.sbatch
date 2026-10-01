#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH -t 03:00:00
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH --job-name=tribeloc_searchlight
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# CPU only, one task per array element (46 tasks):
#   sbatch --array=0-45 slurm/searchlight.sbatch
source slurm/common.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # small matrices: threading only adds overhead
python scripts/04_searchlight.py --task-index "$SLURM_ARRAY_TASK_ID"
