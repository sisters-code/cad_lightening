from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import timm
import torch

from model import Model
from model_multi import ModelMulti
from model_blood_vessel import ModelBloodVessel
from model_multi_part import ModelMultiPart


def _init_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset_1crop_root', type=str, default='data/face_heartDisease_data')
    parser.add_argument('--dataset_4crops_root', type=str, default='/home/aa/data/face_CAD_data/')
    parser.add_argument('--dataset_parts_root', type=str, default='/mnt/hdd/data/henrique/datasets/proc_cad')
    parser.add_argument('--model_name', type=str, default='resnet50', choices=timm.list_models())
    parser.add_argument('--batch_size', type=int, default=32)
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
    parser.add_argument('--replace_conv1', action='store_true')
    parser.add_argument('--use_sam', action='store_true')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--clear_trainer', action='store_true')
    parser.add_argument('--study_name', type=str, default='cad')
    parser.add_argument('--use_multi', action='store_true')
    parser.add_argument('--reduce_fc_channels', type=int, default=0)
    parser.add_argument('--write_predictions', action='store_true')
    parser.add_argument('--use_blood_vessel', action='store_true')
    parser.add_argument('--face_part_name', type=str, default=None)
    parser.add_argument('--face_part_names', type=str, nargs='+', default=None)
    parser.add_argument('--num_parts', type=int, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--aug_train_scales', type=float, nargs='*', default=[])
    parser.add_argument('--use_tta', action='store_true')
    parser.add_argument('--tta_scales', type=float, nargs='*', default=[1.0])
    return parser


def main(args):
    if args.use_multi:
        model = ModelMulti(args)
    elif args.use_blood_vessel:
        model = ModelBloodVessel(args)
    elif args.face_part_names is not None and len(args.face_part_names) > 1:
        model = ModelMultiPart(args)
    else:
        model = Model(args)
    model = model.eval()

    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt['state_dict'])

    if args.log_dir is None:
        args.log_dir = 'lightning_logs_val'
    log_model_dir = str(Path(args.log_dir) / args.model_name)
    if args.face_part_name is not None:
        log_model_dir += f'/{args.face_part_name}'

    tb_logger = pl.loggers.TensorBoardLogger(log_model_dir)

    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, stochastic_weight_avg=args.use_swa)
    trainer.validate(model)


if __name__ == '__main__':
    parser = _init_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
