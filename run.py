import os, csv
import sys
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
from data.cm_spurious_dataset import CifarMnistSpuriousDataset, get_provider
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
    #------------loading Cifar-Mnist Datset---#
    if args.dataset == "SPCM":
        cons_list = [float(x) for x in args.cons_ratios.split("_")]
        train_env_num= len(cons_list) - 1
        ratio_list = [1. / train_env_num] * (train_env_num)
        cifarminist = CifarMnistSpuriousDataset(
            train_num=10000,
            test_num=1800,
            cons_ratios=cons_list,
            train_envs_ratio=ratio_list,
            label_noise_ratio=args.label_noise_ratio,
            color_spurious=0,
            transform_data_to_standard=1,
            oracle=args.oracle)
        train_x, train_y, train_env, train_sp = cifarminist.return_train_data()
        test_x, test_y, test_env, test_sp = cifarminist.return_test_data()
        dp = get_provider(
            batch_size=args.batch_size,
            n_classes=2,
            env_nums=train_env_num,
            train_x=train_x,
            train_y=train_y,
            train_env=train_env,
            train_sp=train_sp,
            train_transform=cifarminist.transform,
            test_x=test_x,
            test_y=test_y,
            test_env=test_env,
            test_sp=test_sp,
            test_transform=cifarminist.transform)
        #---loading Cifar-Mnist Datset Ended---#
    else:
        pass
        """
        you can provide your dataset by the following interface:
        dp = get_provider(
            batch_size=<batch_size>,
            n_classes=<number of classes>,
            env_nums=<number of train envs>,
            train_x=<your_train_x>,
            train_y=<your_train_y>,
            train_env=<your_train_env>,
            train_sp=<your_train_sp>,# optional
            train_transform=<your_train_transform>,# optional
            test_x=<your_test_x>,
            test_y=<your_test_y>,
            test_env=<your_test_env>,
            test_sp=<your_test_sp>,# optional
            test_transform=<your_test_transform> # optional)
        """
    data={}
    data['train_loader'] = dp.train_loader
    data['val_loader'] = dp.test_loader
    data['test_loader'] = dp.test_loader
    data['train_data'] = dp.train_dataset
    data['val_data'] = dp.test_dataset
    data['test_data'] = dp.test_dataset
    n_classes=dp.n_classes
    env_nums = dp.env_nums

    pretrained = args.pretrained
    args.env_nums = train_env
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
    output_path = "results"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    target_file = os.path.join(output_path, 'model.pth')
    print("Outputing file to %s."% target_file)
    # torch.save(model, target_file)
    torch.save({'state_dict': model.state_dict()}, target_file)

if __name__=='__main__':
    main()

