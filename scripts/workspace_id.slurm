#!/bin/bash
#SBATCH --job-name=ws_id
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/workspace_id_%j.out
#SBATCH --error=logs/workspace_id_%j.err

cd /mnt/home/jacob/vl/repos/gspace
source .venv/bin/activate
export PYTHONUNBUFFERED=1

find src/gspace -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find src/gspace -name "*.pyc" -delete 2>/dev/null

python3 -u src/gspace/identify_workspace.py \
    --output-dir results/workspace_id \
    --n-wikitext 20 \
    --max-seq-len 128
