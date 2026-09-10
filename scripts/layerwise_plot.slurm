#!/bin/bash
#SBATCH --job-name=gspace-lwp
#SBATCH --output=logs/layerwise_plot_%j.out
#SBATCH --error=logs/layerwise_plot_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results/layerwise_plot

echo "=== Layerwise G-space vs token-response plot ==="
echo "Started at $(date)"

PYTHONUNBUFFERED=1 python3 -u -m gspace.layerwise_plot \
    --jlens_path /data/jlens/qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt \
    --model_id Qwen/Qwen3.5-27B \
    --activations results/phase1_q35/activations.npz \
    --crosscoder results/crosscoder \
    --output_dir results/layerwise_plot

echo "Finished at $(date)"
