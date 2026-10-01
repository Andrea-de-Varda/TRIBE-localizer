#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH --gres=gpu:a100:1
#SBATCH -t 06:00:00
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --job-name=tribeloc_bench
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Benchmark: smallest shard (17, subject_verb_agreement) and largest (82, number_sorting).
# Writes results/tribe/timing/job_<id>.json; use it to size the array job.
source slurm/common.sh
df -h "${TMPDIR:-/tmp}"
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/03_run_tribe.py --shards 17 82
du -sh results/tribe/shards
