#!/bin/bash
#SBATCH --job-name=gspace-cc
#SBATCH --output=logs/crosscoder_%j.out
#SBATCH --error=logs/crosscoder_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results/crosscoder

echo "=== Crosscoder training: Qwen3.5-27B, mod 59 ==="
echo "Layers: 48-63, Features: 4096, TopK: 32"
echo "Token budget: 50M, Batch: 2048"
echo "Started at $(date)"

# Warmup: ensure model downloads
python3 -c "
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-27B', trust_remote_code=True, 
    torch_dtype='bfloat16', device_map='cpu', max_memory={0: '2GB'})
print('Model cached OK')
"

python3 src/gspace/crosscoder_train.py \
    --modulus 59 \
    --n_features 4096 \
    --k_topk 32 \
    --token_budget 50000000 \
    --batch_size 2048 \
    --fwd_batch 8 \
    --seq_len 128 \
    --layer_start 48 \
    --layer_end 63 \
    --lr 5e-5 \
    --warmup 500 \
    --output_dir results/crosscoder \
    --log_every 200 \
    --save_every 5000

echo "Finished at $(date)"
