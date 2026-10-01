#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH --gres=gpu:a100:1
#SBATCH -t 24:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --job-name=tribeloc_infer
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Full run. Set PER_TASK and the array range from the benchmark, e.g. for 183 shards and 4 per task:
#   sbatch --array=0-45%8 --export=ALL,PER_TASK=4 slurm/inference.sbatch
# Finished shards are skipped, so failed tasks can simply be resubmitted.
source slurm/common.sh
python scripts/03_run_tribe.py --array-index "$SLURM_ARRAY_TASK_ID" --per-task "${PER_TASK:-4}"
