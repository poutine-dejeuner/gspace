#!/bin/bash
#SBATCH --job-name=rot-heads
#SBATCH --output=logs/rotation_heads_%j.out
#SBATCH --error=logs/rotation_heads_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --partition=batch

# Rotation Head Discovery: QK cyclic shift analysis
# Usage:
#   sbatch scripts/rotation_heads.slurm
#   sbatch scripts/rotation_heads.slurm --smoke_test 10   (quick test)

set -e
cd /mnt/home/jacob/vl/repos/gspace
source .venv/bin/activate
mkdir -p logs

echo "========================================="
echo "Rotation Head Discovery"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Args: $@"
echo "========================================="

python -m gspace.rotation_heads "$@"

echo "Done at $(date)"
