# CaLLM: Cascading Autoencoder and Large Language Model for Video Anomaly Detection

## Prerequisites
Create a conda environment with the following packages:
```
conda env create -f environment.yml -n callm
```
Activate the environment:
```
conda activate callm
```

## Dataset Preparation
Video frames from the first two testing clips of CUHK Avenue dataset are used for the fine-tune stage for Video-LLaMA.
The dataset can be downloaded from the following link:
https://drive.google.com/drive/folders/1T1infYNDH91PrDRSWPKnSPkgfzTK-U1z?usp=sharing

## Inference
To run the inference, please follow the steps below:
```
python cascaded.py \
--cfg-path eval_configs/video_llama_eval_only_vl_7B.yaml \
--model_type vicuna \
--gpu-id 0 
```

## Introduction

This project relies on the following GitHub repositories:

- [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA)
- [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)