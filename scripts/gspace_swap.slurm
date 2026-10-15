#!/bin/bash
#SBATCH --job-name=gspace-swap
#SBATCH --output=logs/gspace_swap_%j.out
#SBATCH --error=logs/gspace_swap_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00

echo "========================================="
echo "G-space vs J-space Swap"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Args: $@"
echo "========================================="

source .venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace

python -m gspace.gspace_swap "$@"

echo "Done at $(date)"
