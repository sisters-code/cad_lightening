from argparse import ArgumentParser
from pathlib import Path

import optuna
import pytorch_lightning as pl
import timm
import torch

from model import Model
from model_multi import ModelMulti
from model_blood_vessel import ModelBloodVessel


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
    parser.add_argument('--disable_optuna', action='store_true')
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
    parser.add_argument('--ckpt_path', type=str, default=None)
    return parser


def main(args):
    if args.use_multi:
        model = ModelMulti(args)
    elif args.use_blood_vessel:
        model = ModelBloodVessel(args)
    else:
        model = Model(args)
    model = model.eval()

    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt['state_dict'])

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer.from_argparse_args(args, stochastic_weight_avg=args.use_swa)
    metrics = trainer.validate(model)
    # for k, v in metrics.items():
    #     print(f'{k}: {v:.03f}')


if __name__ == '__main__':
    parser = _init_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
