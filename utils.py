import sys
import datetime
import pandas as pd
from collections import defaultdict
import os
import torch
import numpy as np
import csv

class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.fpath=fpath
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def log_args(args, logger):
    for argname, argval in vars(args).items():
        logger.write(f'{argname.replace("_"," ").capitalize()}: {argval}\n')
    logger.write('\n')

def cmd_generator(para_dict, args, sfx, default_ds_cmd=None):
    name_list = list(para_dict.keys())
    cmd_list = ["python %s" % args.file]
    if default_ds_cmd is not None:
        cmd_list = [x + " " + default_ds_cmd for x in cmd_list]

    def duplicate_cmd(cmd_list, num):
        ncl = []
        for cmd in cmd_list:
            for i in range(num):
                ncl.append(cmd)
        return ncl

    for iname in name_list:
        ilist = para_dict[iname]
        cmd_list = duplicate_cmd(
            cmd_list, len(ilist))
        for icmd in range(len(cmd_list)):
            value = ilist[icmd % len(ilist)]
            if isinstance(value, bool):
                if value == True:
                    cmd_list[icmd] = cmd_list[icmd] + " --%s" % iname
                elif value == False:
                    pass
            else:
                cmd_list[icmd] = cmd_list[icmd] + " --%s %s" % (iname, value)

# CUDA_list = [int(x) for x in args.gpu.split(",")]
# print("CUDA_list", CUDA_list)
    import random
    random.shuffle(cmd_list)
    print("cmd number=%s" % len(cmd_list))
    CUDA_num = args.gpu_num
    sh_dict = defaultdict(list)
    for icmd in range(len(cmd_list)):
        cuda_device = icmd % CUDA_num
        sh_dict[cuda_device].append(cmd_list[icmd])

    run_sh_cmds = []

    dir_path = "meta_run/auto/%s" % sfx
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    import json
    exDict = para_dict
    exDict["time"] = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    with open(dir_path + "/args", 'w') as file:
        file.write(json.dumps(exDict))
    print("args json to", dir_path + "/args")

    for i in range(CUDA_num):
        sh_cmd = sh_dict[i]
        file_name = '%s/%s.sh' % (dir_path, i)
        print("sh  %s"% file_name)
        run_sh_cmds.append("sh " + file_name)
        with open(file_name, 'w') as f:
            for item in sh_cmd:
                f.write("%s\n" % item)
            if args.add:
                f.write("sh  meta_run/auto/SPCelebA__adv__step_______________fake_____________EWloBe/0.sh")

    with open(os.path.join(dir_path, "cmd"), "w") as f:
        f.write('python ' + ' '.join(sys.argv))
        print("Outputing cmd.")

def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

class EpochStat(object):
    def __init__(self, sfx):
        self.attrs = []
        self.sfx = sfx

    def update_stats(self, stats):
        if isinstance(stats, list):
            for es in stats:
                self.attrs.append(es)
        elif isinstance(stats, dict):
            self.attrs.append(stats)
        else:
            raise Exception

    def get_summary(self):
        summary_dict = {}
        full_df = pd.DataFrame(self.attrs)
        self.envs = list(np.unique(full_df.env))
        for ie in self.envs:
            edf = full_df[full_df.env == ie]
            summary_dict[ie] = dict(edf.mean())
        self.summary_dict = summary_dict
        return summary_dict

    def get_log_summary(self):
        self.get_summary()
        log_dict = {}
        for ie in self.envs:
            env_dict = self.summary_dict[ie]
            log_dict["avg_acc_group:%s"%ie] = env_dict["acc"]
            log_dict["avg_loss_group:%s"%ie] = env_dict["loss"]
        self.log_dict = log_dict
        return log_dict


    def echo(self):
        summary_dict = self.get_summary()
        echo_str = [self.sfx + "\n"]
        for ienv in range(len(self.envs)):
            env = self.envs[ienv]
            echo_str += ["Env%s"%env]
            for k,v in summary_dict[env].items():
                if k != "env" and not np.isnan(v):
                    echo_str += ["%s=%.4f"%(k, v)]
            if ienv < len(self.envs) - 1:
                echo_str += ["\n"]
        print(" ".join(echo_str))

def env_stat(x, outputs, y, g, model, criterion):
    env_stats = []
    for ie in range(4):
        eindex = (g == ie)
        if eindex.sum() > 0:
            ex = x[eindex]
            ey = y[eindex]
            env_stats.append(
            {"env": ie,
            "loss": criterion(
                outputs[eindex].view(-1),
                ey.float()).item(),
            "acc": mean_accuracy(
                outputs[eindex].view(-1),
                ey.float()).item()})
    return env_stats


def append_tvt(kv_dict, header=None):
    assert header is not None
    newkv = {}
    for k, v in kv_dict.items():
        newkv.update({header+"_" + k: v})
    return newkv

def Tensor2Dict(tensor, name):
    np_tensor = tensor.detach().cpu().numpy()
    list_tensor = np_tensor.ravel().tolist()
    list_name = [name+"_"+str(i) for i in range(len(list_tensor))]
    return dict(zip(list_name, list_tensor))

def next_batch(loader_iter, loader, dataset, batch_size=None):
    if batch_size is not None:
        if batch_size >= 0.95* len(loader.dataset.x_array):
           ds = loader.dataset
           batch_data = \
                torch.Tensor(ds.x_array).float(), \
                torch.Tensor(ds.y_array).float(), \
                torch.Tensor(ds.env_array)
           return batch_data, loader_iter
    try:
        batch_data = loader_iter.__next__()
    except:
        loader_iter = iter(loader)
        batch_data = loader_iter.__next__()
    if dataset in ["PACS", "VLCS", "office_home"]:
        batch_list = batch_data
        try:
            images = torch.cat([i[0] for i in batch_list], dim=0)
            target = torch.cat([i[1] for i in batch_list], dim=0)
            domain_idx = torch.cat([i[2] for i in batch_list], dim=0)
        except:
            print(batch_list[0][0].shape, batch_list[0][1], batch_list[0][2], len(batch_list))
            raise Exception
        batch_data = (images, target, domain_idx)
    return batch_data, loader_iter


def env_average(samples, env):
    if len(list(samples.size())) == 1:
        samples = samples.view(
            len(samples),
            -1)
    samples = samples
    labels = env.long()
    M = torch.zeros(
        labels.max().long().item()+1,
        len(samples)).cuda()
    M[labels, torch.arange(len(samples))] = 1
    M = torch.nn.functional.normalize(M, p=1, dim=1)
    return torch.mm(M, samples)


def convert_fmt(df, flds):
    out_df = df
    format_result = []
    for ir in range(out_df.shape[0]):
        count = -1
        one_dict = {"model": out_df.iloc[ir]["model"][0]}
        for ifd in flds: # ["best_test", "best_train", 'best_loss', 'best_loc']:
            if count == -1:
                count =  out_df.iloc[ir][(ifd, "count")]
                one_dict.update({"count": count})
            if ifd != 'best_loc':
                meanifd = out_df.iloc[ir][(ifd, "mean")] * 100 
                stdifd = out_df.iloc[ir][(ifd, "std")] * 100
            else:
                meanifd = out_df.iloc[ir][(ifd, "mean")]  
                stdifd = out_df.iloc[ir][(ifd, "std")]
            if np.isnan(stdifd):
                format_ifd = ("%.2f" % meanifd)
            else:
                format_ifd = ("$%.2f \pm %.2f$" % (meanifd , stdifd))
            one_dict.update({ifd: format_ifd})
        format_result.append(one_dict)
    format_df = pd.DataFrame(format_result)
    return format_df
