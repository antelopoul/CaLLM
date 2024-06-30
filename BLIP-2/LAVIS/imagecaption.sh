#!/bin/bash
#SBATCH --job-name=blip2
#SBATCH --partition prioritized
#SBATCH --nodelist=a256-a40-06
#SBATCH --time 12:00:00
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=4
#SBATCH --qos=normal
#SBATCH --output=blip2_img_caption.out
#SBATCH --error=blip2_img_caption.err
srun -G1 singularity run --nv ~/sandboxes/blip2 python image_caption.py \
    --folder_path ~/projects/VAD/LearningNotToReconstructAnomalies/dataset/ped2/testing/frames/12 \
    --output_path ./captions/ped2/12.json \