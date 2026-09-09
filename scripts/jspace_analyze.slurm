#!/bin/bash
#SBATCH --job-name=gspace-jsp
#SBATCH --output=logs/jspace_analyze_%j.out
#SBATCH --error=logs/jspace_analyze_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results/jspace_analysis

echo "=== J-space analysis ==="
echo "Started at $(date)"

PYTHONUNBUFFERED=1 python3 -u -m gspace.jspace_analyze \
    --jlens_path /data/jlens/qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt \
    --model_id Qwen/Qwen3.5-27B \
    --activations results/phase1_q35/activations.npz \
    --crosscoder results/crosscoder \
    --output_dir results/jspace_analysis \
    --k_sparse 25

echo "Finished at $(date)"
