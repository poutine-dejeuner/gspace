#!/bin/bash
#SBATCH --job-name=meas-rot
#SBATCH --output=logs/measure_rotation_%j.out
#SBATCH --error=logs/measure_rotation_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

echo "========================================="
echo "Inter-layer Rotation Measurement"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Args: $@"
echo "========================================="

source .venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace

python -m gspace.measure_rotation "$@"

echo "Done at $(date)"
