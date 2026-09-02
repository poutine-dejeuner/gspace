#!/bin/bash
#SBATCH --job-name=gspace-p0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=results/phase0_%j.out
#SBATCH --error=results/phase0_%j.err

set -e
cd /mnt/home/jacob/vl/repos/gspace

source .venv/bin/activate

echo "=== gspace Phase 0: Model Selection ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

python -m gspace.phase0_benchmark

echo "=== Done: $(date) ==="
