#!/bin/bash
#SBATCH -p pi_evelina9
#SBATCH --gres=gpu:a100:1
#SBATCH -t 02:00:00
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --job-name=braingpt_score
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# BrainGPT perplexity scoring of all abstracts (one GPU), then the summary (CPU). Run from the repository root:
#   sbatch braingpt/slurm/score.sbatch
# Needs Hugging Face access to mistralai/Mistral-7B-v0.1 (gated) and, once, in the tribe env:
#   pip install peft accelerate --no-deps     (--no-deps keeps the TRIBE package versions unchanged)
source slurm/common.sh
set -e                                       # stop at the first failing step
python -c "import peft, accelerate; print('peft', peft.__version__, 'accelerate', accelerate.__version__)"
cd braingpt
python score.py
python analyze.py
python plot_assignments.py
