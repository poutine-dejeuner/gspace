#!/bin/bash
#SBATCH --job-name=np-swap
#SBATCH --output=logs/neuronpedia_swap_%j.out
#SBATCH --error=logs/neuronpedia_swap_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

echo "========================================="
echo "Neuronpedia-style J-lens Swap"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Args: $@"
echo "========================================="

source .venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace

python -m gspace.neuronpedia_swap "$@"

echo "Done at $(date)"
