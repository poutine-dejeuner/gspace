#!/bin/bash
#SBATCH --job-name=gspace-causal
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gspace_causal_%j.out
#SBATCH --error=logs/gspace_causal_%j.err

cd /mnt/home/jacob/vl/repos/gspace
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# Clear stale __pycache__
find src/gspace -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find src/gspace -name "*.pyc" -delete 2>/dev/null

python3 -u src/gspace/gspace_causal.py \
    --experiments 1,2,3,4 \
    --output-dir results/gspace_causal
