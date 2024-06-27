#!/usr/bin bash
#SBATCH --job-name VLLaMA
#SBATCH --nodelist=a256-a40-07
#SBATCH --partition prioritized
#SBATCH --time 12:00:00
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
srun --pty singularity run ~/singularities/vad2501.sif infer.py\
    --model openllama_peft\
    --cfg-path eval_configs/video_llama_eval_only_vl.yaml\
    --model_type llama_v2\
    --gpu-id 0\
    --video_input examples/skateboarding_dog.mp4\
    --prompt "What is the dog doing?"