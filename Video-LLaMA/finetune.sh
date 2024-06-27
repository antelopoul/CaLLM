#!/bin/bash
#SBATCH --job-name=VLLAMA
#SBATCH --partition batch
#SBATCH --time 12:00:00
#SBATCH --gres=gpu:8
#SBATCH --nodelist=a768-l40-01
#SBATCH --mem=320G
#SBATCH --qos=normal
#SBATCH --output=vllama.out
#SBATCH --error=vllama.err
srun -G8 singularity run --nv ~/singularities/vad2501.sif torchrun train.py\
    --cfg-path ./train_configs/visionbranch_stage2_finetune.yaml\
    
echo "Job completed"
