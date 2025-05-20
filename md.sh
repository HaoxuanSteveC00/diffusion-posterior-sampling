#!/bin/bash

#SBATCH --partition prod
#SBATCH --job-name=md
#SBATCH --output=/home/ml/zach/diffusion-posterior-sampling/log/md.out
#SBATCH --error=/home/ml/zach/diffusion-posterior-sampling/log/md.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPUMODEL_A6000


######################
# Begin work section #
######################
conda init
conda activate bmposterior
python3 sample_condition.py \
    --model_config=configs/model_imagenet_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/motion_deblur_imagenet_config.yaml \
    --gpu=0 \
    --save_dir=results/;