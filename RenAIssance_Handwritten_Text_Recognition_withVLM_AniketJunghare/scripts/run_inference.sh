#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --time=6-00:00:00
#SBATCH --output=inference_on_given_test_images.%j.out
#SBATCH --error=inference_on_given_test_images.%j.err


# Load conda and activate env
source /home/vis-comp/aniketjunghare/anaconda3/etc/profile.d/conda.sh
conda activate GSOC_26

# Run inference
python -u inference.py \
  --model-dir models/Qwen2.5-VL-7B-Instruct \
  --lora models/qwen2.5-vl-ocr-lora-handwritten/final \
  --image-dir data/test_images_handwritten \
  --output-visual results/Visual_Results 
  --trocr models/mim-trocr-gsoc25
  
