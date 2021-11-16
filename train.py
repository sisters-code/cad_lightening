from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import timm
import torch

from model import ModelMultiPart
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'

def _init_parser():
    parser = ArgumentParser()
    parser.add_argument('--expid', type=str, default='debug')
    parser.add_argument('--dataset_4crops_root', type=str, default='/home/aa/data/face_CAD_data/')
    parser.add_argument('--dataset_parts_root', type=str, default='/home/aa/data/proc_cad')
    parser.add_argument('--model_name', type=str, default='resnet50', choices=timm.list_models())
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--square_resize', action='store_true')
    parser.add_argument('--crop_size', type=int, nargs=2, default=(256, 256))
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--multistep', type=int, nargs='+', default=(50, 90,))
    parser.add_argument('--step_gamma', type=float, default=0.1)
    parser.add_argument('--freeze_epoch', type=int, default=1000)
    parser.add_argument('--balance_samples', action='store_true')
    parser.add_argument('--freeze_bn', action='store_true')
    parser.add_argument('--use_swa', action='store_true')
    parser.add_argument('--backbone_lr_multiplier', type=float, default=1.0)
    parser.add_argument('--image_names', type=str, nargs='+', default=('1.JPG',))
    parser.add_argument('--use_sam', action='store_true')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--clear_trainer', action='store_true')
    parser.add_argument('--reduce_fc_channels', type=int, default=512)
    parser.add_argument('--write_predictions', action='store_true')
    parser.add_argument('--face_part_names', type=str, nargs='+', default=None)
    parser.add_argument('--num_parts', type=int, default=None)
    parser.add_argument('--aug_train_scales', type=float, nargs='*', default=[])
    parser.add_argument('--use_tta', action='store_true')
    parser.add_argument('--tta_scales', type=float, nargs='*', default=[1.0])
    parser.add_argument('--use_gray', action='store_true')
    parser.add_argument('--use_eqhist', action='store_true')
    parser.add_argument('--use_gamma', action='store_true')
    return parser


def main(args):
    model = ModelMultiPart(args)
    model = model.train()

    if args.resume_from_checkpoint is not None and args.clear_trainer:
        ckpt = torch.load(args.resume_from_checkpoint)
        model.load_state_dict(ckpt['state_dict'])
        args.resume_from_checkpoint = None

    if args.log_dir is None:
        args.log_dir = 'lightning_logs'
    log_model_dir = str(Path(args.log_dir) / args.model_name)

    tb_logger = pl.loggers.TensorBoardLogger(log_model_dir, name=args.expid)

    model_ckpt_last = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filename=args.model_name+'_last_{epoch}_{step}', save_weights_only=True)
    model_ckpt_train = pl.callbacks.model_checkpoint.ModelCheckpoint(filename=args.model_name+'_train_{epoch}_{step}')
    model_ckpt_best = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filename=args.model_name+'_best_{auroc:.2f}_{epoch}_{step}', save_weights_only=True,
        save_top_k=1, monitor='auroc', mode='max')

    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, auto_select_gpus=True, stochastic_weight_avg=args.use_swa, callbacks=[model_ckpt_last, model_ckpt_train, model_ckpt_best])
    trainer.fit(model)


if __name__ == '__main__':
    parser = _init_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
