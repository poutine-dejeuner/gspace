#!/bin/bash
#SBATCH --job-name=gs-finetune
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace

mkdir -p logs

echo "Starting fine-tuning at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

python -m gspace.finetune 2>&1

echo "Done at $(date)"
