#/bin/bash

# $1: task
# $2: gpu number


# FFHQ

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/gaussian_deblur_config.yaml \
    --gpu=0 \
    --save_dir=results/;

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/motion_deblur_config.yaml \
    --gpu=0 \
    --save_dir=results/;

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/super_resolution_config.yaml \
    --gpu=0 \
    --save_dir=results/;

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/inpainting_config.yaml \
    --gpu=0 \
    --save_dir=results/;   


# ImageNet

python3 sample_condition.py \
    --model_config=configs/model_imagenet_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/gaussian_deblur_imagenet_config.yaml \
    --gpu=0 \
    --save_dir=results/; 

python3 sample_condition.py \
    --model_config=configs/model_imagenet_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/motion_deblur_imagenet_config.yaml \
    --gpu=0 \
    --save_dir=results/;

python3 sample_condition.py \
    --model_config=configs/model_imagenet_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/super_resolution_imagenet_config.yaml \
    --gpu=0 \
    --save_dir=results/;

python3 sample_condition.py \
    --model_config=configs/model_imagenet_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/inpainting_imagenet_config.yaml \
    --gpu=0 \
    --save_dir=results/;        


