#!/bin/bash
cmd="python validate.py
    --model_name $1
    --face_part_names face left_ear right_ear
    --num_parts 2
    --gpus 1
    --use_tta
    --tta_scales 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5
    --log_dir multi_parts_logs_val"

if [ ! -z $2 ]
    then
        cmd="${cmd} --ckpt_path $2"
fi

echo ${cmd}
eval ${cmd}