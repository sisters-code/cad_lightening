#!/bin/bash
cmd="CUDA_VISIBLE_DEVICES=1 python train.py
    --expid base_line_mp_noRPRA
    --model_name resnet18
    --batch_size 32
    --face_part_names face right_ear left_ear
    --aug_train_scales 1.0
    --num_parts 2
    --gpus 1
    --log_dir multi_parts_logs
    --lr 0.001
    --wd 0.0
    --balance
    --max_epochs 50
    --multistep 10 30
    --use_gamma
    --use_sam
    --use_swa
    --freeze_bn
    "

#    --use_sam
#    --use_swa
#    --freeze_bn
echo ${cmd}
eval ${cmd}
