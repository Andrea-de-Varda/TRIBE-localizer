#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH --gres=gpu:a100:1
#SBATCH -t 06:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --job-name=tribeloc_infer
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Full run. Benchmark: the largest shard (75k words) took 14 min on an A100, so 4 shards per task
# take at most about 1 h; the whole run is about 20 GPU-hours. For 183 shards and 4 per task:
#   sbatch --array=0-45%8 --export=ALL,PER_TASK=4 slurm/inference.sbatch
# Finished shards are skipped, so failed tasks can simply be resubmitted.
source slurm/common.sh
python scripts/03_run_tribe.py --array-index "$SLURM_ARRAY_TASK_ID" --per-task "${PER_TASK:-4}"
