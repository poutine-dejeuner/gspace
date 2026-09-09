#!/bin/bash
#SBATCH --job-name=gspace-cca
#SBATCH --output=logs/cc_analyze_%j.out
#SBATCH --error=logs/cc_analyze_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results/crosscoder_analysis

echo "=== Crosscoder analysis ==="
echo "Started at $(date)"

python3 src/gspace/crosscoder_analyze.py \
    --crosscoder results/crosscoder \
    --activations results/phase1_q35/activations.npz \
    --output_dir results/crosscoder_analysis

echo "Finished at $(date)"
