# CaLLM: Cascading Autoencoder and Large Language Model for Video Anomaly Detection

## Introduction

This work proposes the integration of a weak classifier and a Visual-Language Model Video-LLaMA for the video anomaly detection and explanation of a video event from surveiliance footage. It is built on top [MemAE] (https://github.com/donggong1/memae-anomaly-detection) and [Video-LLaMA] (https://github.com/DAMO-NLP-SG/Video-LLaMA). We use only the Vision-Language Branch using the pre-training data on WebVid (2.5M video-caption pairs) and LLaVA-CC3M (595k image-caption pairs) and fine-tune with weak-label captions generated from two testing clips of CUHK Avenue dataset. 
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

Config the checkpoint in Video-LLaMA and select the appropriate dataset paths in Video-LLaMA/eval_configs/video_llama_eval_only_vl_7B.yaml.

```
python cascaded.py \
--cfg-path eval_configs/video_llama_eval_only_vl_7B.yaml \
--model_type vicuna \
--gpu-id 0 
```


This project relies on the following GitHub repositories:

- [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA)
- [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)