#!/bin/bash
#SBATCH --job-name=VLLAMA
#SBATCH --partition=prioritized
#SBATCH --nodelist=nv-ai-03
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-gpu=6
#SBATCH --qos=normal
#SBATCH --output=evalfinetuned/avenue.out

srun -G1 singularity run --nv ~/singularities/vad2501.sif python cascaded.py \
    --cfg-path eval_configs/video_llama_eval_only_vl_7B.yaml \
    --model_type vicuna --gpu-id 0

echo "Job completed"
