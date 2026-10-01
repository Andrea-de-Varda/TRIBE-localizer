#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH --gres=gpu:a100:1
#SBATCH -t 06:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --job-name=tribeloc_infer
#SBATCH --array=0-45%6          # 46 tasks x 4 shards = 183 shards; at most 6 GPUs at a time
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Full run. Benchmark: the largest shard (75k words) took 14 min on an A100, so 4 shards per task
# take at most about 1 h; the whole run is about 20 GPU-hours (about 4 h wall time with 6 GPUs).
#   sbatch slurm/inference.sbatch
# Never run more than 6 GPUs at a time (lab policy); keep the %6 if overriding --array.
# Finished shards are skipped, so failed tasks can simply be resubmitted.
source slurm/common.sh
python scripts/03_run_tribe.py --array-index "$SLURM_ARRAY_TASK_ID" --per-task "${PER_TASK:-4}"
