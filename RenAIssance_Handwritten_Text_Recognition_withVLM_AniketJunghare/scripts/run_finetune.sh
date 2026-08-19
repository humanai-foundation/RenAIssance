#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --time=6-00:00:00
#SBATCH --output=vlm_finetune.%j.out
#SBATCH --error=vlm_finetune.%j.err

# Load conda and activate env
source /home/vis-comp/aniketjunghare/anaconda3/etc/profile.d/conda.sh
conda activate GSOC_26

# Run finetuning
python -u finetune.py \
  --model-dir models/Qwen2.5-VL-7B-Instruct \
  --image-dir data/Handwriting-scans \
  --gt-dir data/Handwriting-transcriptions \
  --output-dir models/qwen2.5-vl-ocr-lora-handwritten
