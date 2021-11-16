#!/bin/bash
cmd="python test.py
    --model_name $1
    --gpus 1
    --use_swa
    --image_names 1.JPG
    --write_predictions
    --ckpt $2"
    # --image_names 1.JPG"
    # --replace_conv1
    # --dropout 0.5
    # --reduce_fc_channels 512

echo ${cmd}
eval ${cmd}