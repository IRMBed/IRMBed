import os, csv
import sys
import wandb
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from resnet_ofc import resnet18_invrat_ec,resnet50_invrat_ec
from resnet_ofc import resnet18_invrat_eb, resnet50_invrat_eb
from resnet_ofc import resnet50_irmgame, resnet18_irmgame
# resnet18 is the same structre with resnet18_invrat_ec
from resnet_ofc import resnet18_invrat_ec as resnet18
from resnet_ofc import resnet50_invrat_ec as resnet50
from data.cm_spurious_dataset import get_data_loader_cifarminst
from utils import set_seed, Logger, log_args
from train import train


def main():
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument(
        '--model', default='resnet18')
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('--val_percentage', default=0.1, type=float)
    parser.add_argument('--cons_ratios', default='0.9,0.8,0.1', type=str)
    parser.add_argument('--train_envs_ratio', default='-1', type=str)
    parser.add_argument('--label_noise_ratio', default=0, type=float)
    parser.add_argument('--step_gamma', default=0.2, type=float)
    parser.add_argument('--oracle', default=0, type=int, choices=[0, 1])
    parser.add_argument('--irm_penalty', default=False, action='store_true')
    parser.add_argument('--irm_type', default="irmv1", choices=["invrat", "irmgame", "irmv1", "irmv1fc", "rex", "erm", "rvp"], type=str)
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
        print("WARNING:Using ORACLE dataset!!!!!")
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
        transform_data_to_standard=1,
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
    if args.model == 'resnet18':
        model = resnet18(pretrained=pretrained, num_classes=n_classes)
    elif args.model == 'resnet50':
        model = resnet50(pretrained=pretrained, num_classes=n_classes)
    elif args.model == 'resnet18_invrat_ec':
        model = resnet18_invrat_ec(pretrained=pretrained, num_classes=n_classes)
    elif args.model == 'resnet50_invrat_ec':
        model = resnet50_invrat_ec(pretrained=pretrained, num_classes=n_classes)
    elif args.model == 'resnet18_irmgame':
        model = resnet18_irmgame(pretrained=pretrained, num_classes=n_classes)
    elif args.model == 'resnet50_irmgame':
        model = resnet50_irmgame(pretrained=pretrained, num_classes=n_classes)
    elif args.model == "resnet18_invrat_eb":
        model = resnet18_invrat_eb(pretrained=pretrained, num_classes=n_classes)
    elif args.model == "resnet50_invrat_eb":
        model = resnet50_invrat_eb(pretrained=pretrained, num_classes=n_classes)
    else:
        raise ValueError('Model not recognized.')


    train(model, data, None, args)

if __name__=='__main__':
    main()

