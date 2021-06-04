import os, csv
import sys
import wandb
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from models import model_attributes, ModelFeature, MLP2, get_norm_laryer
from resnet_ofc import resnet18_sepfc_ofc, resnet18_sepslc_ofc, resnet50_sepslc_ofc, resnet50_sepfc_ofc
from resnet_ofc import resnet18_sep2fc_invrat, resnet50_sep2fc_invrat
from resnet_ofc import resnet50_sepfc_game, resnet18_sepfc_game
from data.cm_spurious_dataset import get_data_loader_cifarminst
from utils import set_seed, Logger, LYCSVLogger, CSVBatchLogger, log_args
from ntrain_irmv12 import train


def main():
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('--val_percentage', default=0.1, type=float)
    parser.add_argument('--cons_ratios', default='0.9,0.8,0.1', type=str)
    parser.add_argument('--train_envs_ratio', default='-1', type=str)
    parser.add_argument('--label_noise_ratio', default=0, type=float)
    parser.add_argument('--step_gamma', default=0.2, type=float)
    parser.add_argument('--clear_momentum', type=int, choices=[0, 1], default=1)
    parser.add_argument('--oracle', default=0, type=int, choices=[0, 1])
    parser.add_argument('--irm_penalty', default=False, action='store_true')
    parser.add_argument('--positive_constrain', default=1, type=int, choices=[0, 1])
    parser.add_argument('--balance_share_sep_batch', default=0, type=int, choices=[0, 1])
    parser.add_argument('--irm_type', default="irmv1", choices=["stepfree", "stepgame", "irmv1", "rex", "erm", "rvp"], type=str)
    parser.add_argument('--penalty_wlr', type=float, default=1.0)
    parser.add_argument('--penalty_welr', type=float, default=1.0)
    parser.add_argument('--lr_schedule_type', type=str, default="step")
    parser.add_argument('--opt', type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument('--num_inners', type=int, default=1)
    parser.add_argument('--irm_penalty_weight', type=float, default=0.0)
    parser.add_argument('--irm_anneal_epochs', type=int, default=0.0)
    parser.add_argument('--irm_anneal_type', type=str, default="jump", choices=["jump", "linear"])

    parser.add_argument('--pretrained', type=int, default=1)

    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.oracle:
        print("LYWARNING:Using ORACLE dataset!!!!!")
    get_loader_x = get_data_loader_cifarminst
    train_num=10000
    test_num=1800
    if args.train_envs_ratio == "-1":
        cons_list = [float(x) for x in args.cons_ratios.split("_")]
        train_envs = len(cons_list) - 1
        ratio_list = [1. / train_envs] * (train_envs)
    else:
        ratio_list = [float(x) for x in args.train_envs_ratio.split("_")]
    args.env_nums = len(ratio_list)
    spd, train_loader, val_loader, test_loader, train_data, val_data, test_data = get_loader_x(
        batch_size=args.batch_size,
        train_num=train_num,
        test_num=test_num,
        cons_ratios=[float(x) for x in args.cons_ratios.split("_")],
        train_envs_ratio=ratio_list,
        label_noise_ratio=args.label_noise_ratio,
        color_spurious=0,
        transform_data_to_standard=1
        oracle=args.oracle)
    data={}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data

    n_classes=spd.n_classes
    feature_dim = spd.feature_dim
    args.env_nums = spd.n_train_envs


    pretrained = args.pretrained
    if args.model == 'resnet18_sepfc_ofc':
        model = resnet18_sepfc_ofc(pretrained=pretrained, num_classes=n_classes)
    elif args.model == 'resnet18_sepslc_ofc':
        model = resnet18_sepslc_ofc(pretrained=pretrained, num_classes=n_classes)
    elif args.model == 'resnet50_sepfc_ofc':
        model = resnet50_sepfc_ofc(pretrained=pretrained, num_classes=n_classes)
    elif args.model == 'resnet50_sepfc2_ofc':
        model = resnet50_sepfc2_ofc(pretrained=pretrained, num_classes=n_classes)
    elif args.model == 'resnet18_sepfc2_ofc':
        model = resnet18_sepfc2_ofc(pretrained=pretrained, num_classes=n_classes)
    elif args.model == 'resnet18_sepfc_game':
        model = resnet18_sepfc_game(pretrained=pretrained, num_classes=n_classes)
    elif args.model == 'resnet50_sepfc_game':
        model = resnet50_sepfc_game(pretrained=pretrained, num_classes=n_classes)
    elif args.model == 'resnet50_sepslc_ofc':
        model = resnet50_sepslc_ofc(pretrained=pretrained, num_classes=n_classes)
    elif args.model == "resnet18_sep2fc_invrat":
        model = resnet18_sep2fc_invrat(pretrained=pretrained, num_classes=n_classes)
    elif args.model == "resnet50_sep2fc_invrat":
        model = resnet50_sep2fc_invrat(pretrained=pretrained, num_classes=n_classes)
    else:
        raise ValueError('Model not recognized.')


    train(model, data, None, train_csv_logger, val_csv_logger, test_csv_logger, args, logger_path, epoch_offset=epoch_offset)

if __name__=='__main__':
    main()

