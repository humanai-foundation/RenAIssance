#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --time=6-00:00:00
#SBATCH --output=rodrigo_evaluation.%j.out
#SBATCH --error=rodrigo_evaluation.%j.err


# Load conda and activate env
source /home/vis-comp/aniketjunghare/anaconda3/etc/profile.d/conda.sh
conda activate GSOC_26

# Run evaluation
python -u evaluate.py \
  --model-dir models/Qwen2.5-VL-7B-Instruct \
  --lora models/qwen2.5-vl-ocr-lora-handwritten/final \
  --image-dir data/Rodrigo_eval/Rodrigo_Images \
  --gt-dir data/Rodrigo_eval/Rodrigo_Transcriptions \
  --output-csv eval_results/Rodrigo_evaluation_results.csv
