import os
import random
import pandas as pd
import pdb
import wandb
import torch.autograd as autograd
import torch
from collections import OrderedDict
from collections import defaultdict
import collections
import types
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np

import torch.optim as optim



def balance_share_batch(x, y, g, args):
    sample_locs = torch.rand(len(x)) <= 1. / args.env_nums
    if sample_locs.sum() > 1:
        x_sam, y_sam, g_sam = x[sample_locs], y[sample_locs], g[sample_locs]
        return x_sam, y_sam, g_sam
    else:
        return x, y, g

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
            try:
                log_dict["penalty:%s"%ie] = env_dict["penalty"]
                log_dict["total_loss:%s"%ie] = env_dict["loss"]
                log_dict["opt_loss:%s"%ie] = env_dict["main_loss"]
                log_dict["major_acc:%s"%ie] = env_dict["major_acc"]
                log_dict["minor_acc:%s"%ie] = env_dict["minor_acc"]

            except:
                pass
        self.log_dict = log_dict
        return log_dict


    def echo(self):
        summary_dict = self.get_summary()
        echo_str = [self.sfx + "\n"]
        for ienv in range(len(self.envs)):
            env = self.envs[ienv]
            if env == '-1':
                echo_str += ["All Envs"]
            else:
                echo_str += ["Env%s"%env]
            for k,v in summary_dict[env].items():
                if k != "env" and not np.isnan(v):
                    echo_str += ["%s=%.4f"%(k, v)]
            if ienv < len(self.envs) - 1:
                echo_str += ["\n"]
        print(" ".join(echo_str))

def env_stat(x, outputs, y, g, model, criterion, sp=None):
    # sp denotes the positinve or negative of spurious correlation
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
            if sp is not None:
                env_stats[-1]["major_acc"] = mean_accuracy(
                        outputs[eindex & (sp==1)].view(-1),
                        y[eindex & (sp==1)].float()).item()
                env_stats[-1]["minor_acc"] = mean_accuracy(
                        outputs[eindex & (sp==0)].view(-1),
                        y[eindex & (sp==0)].float()).item()


    return env_stats


# calculate IRM penalty and main loss
def irm_main_loss(model, x, y, g, criterion):
    model.set_sep(False, key="penalty")
    outputs = model(x)
    individual_loss = criterion(
        outputs.view(-1), y.float())

    model.set_sep(True)
    sep_outputs = model(x, group_idx=g)
    sep_loss = criterion(
        sep_outputs.view(-1), y.float())

    penalty_loss = individual_loss - sep_loss
    model.set_sep(False)
    return individual_loss, penalty_loss


def irm_inner_loss(model, x, y, g, criterion):
    inner_sep_outputs = model(x, group_idx=g, only_backward_fc=True)
    inner_individual_loss = criterion(
        inner_sep_outputs.view(-1), y.float())
    return inner_individual_loss


def irm_env_inner_loss(model, x, y, g, criterion):
    inner_sep_outputs = model.forward_env(x, y_idx=y, only_backward_fc=True)
    inner_individual_loss = nn.CrossEntropyLoss()(
        inner_sep_outputs, g.long())
    return inner_individual_loss

# calculate IRM penalty and main loss
def irm_env_main_loss(model, x, y, g, criterion):
    criterion = nn.CrossEntropyLoss()
    model.set_sep(False)
    outputs = model.forward_env(x)
    individual_loss = criterion(
        outputs, g.long())
    model.set_sep(True)
    sep_outputs = model.forward_env(x, y_idx=y)
    sep_loss = criterion(
        sep_outputs, g.long())

    penalty_loss = individual_loss - sep_loss
    model.set_sep(False)
    return individual_loss, penalty_loss


def compute_irm_free_same_step_penalty_loss(y, group_idx, model, xyg, optimizer, loader, inner_loader_iter, criterion, args):
    x0, y0, g0 = xyg
    for batch_idx_inner in range(args.num_inners):
        batch, inner_loader_iter = next_batch(inner_loader_iter, loader, args.dataset)
        batch = tuple(t.cuda() for t in batch)
        x = batch[0]
        y = batch[1]
        g = batch[2]
        model.set_sep(True)
        inner_batch_loss = irm_inner_loss(model, x, y, g, criterion)
        step_sep(inner_batch_loss, optimizer, model, args)
        model.set_sep(False)
    loss, penalty= irm_main_loss(model, x0, y0, g0, criterion)
    penalty = 4.0 * penalty / (512 * args.lr * args.penalty_welr)
    if penalty < 0:
        penalty = torch.tensor(0.0)
    return loss, penalty, inner_loader_iter


def compute_irm_game_free_same_step_penalty_loss(y, group_idx, model, xyg, optimizer, loader, inner_loader_iter, criterion, args):
    # begin to run sep
    x0, y0, g0 = xyg
    for batch_idx_inner in range(args.num_inners):
        batch, inner_loader_iter = next_batch(inner_loader_iter, loader, args.dataset)
        batch = tuple(t.cuda() for t in batch)
        x = batch[0]
        y = batch[1]
        g = batch[2]
        for j in range(args.env_nums):
            gindex = g == j
            ex = x[gindex]
            ey = y[gindex]
            eg = g[gindex]
            optimizer[j+1].zero_grad()
            inner_batch_loss = irm_inner_loss(
                model, ex, ey, eg, criterion)
            backward_loss(inner_batch_loss, model, grad_clip=-1, retain_graph=True)
            optimizer[j+1].step()
            optimizer[j+1].zero_grad()
    loss, penalty = torch.tensor(0.0), torch.tensor(0.0)
    return loss, penalty, inner_loader_iter


def _irm_penalty_v1(logits, y, criterion):
    # need to see the dimension of logits[::2]
    if logits.shape[0] == 0:
        return 0, 0
    p = torch.randperm(len(logits))
    logits = logits[p]
    y = y[p]

    scale = torch.tensor(1.).cuda().requires_grad_()
    if logits.shape[0] >= 2:
        loss_1 = criterion(
            (logits * scale)[0::2].view(-1),
            y.float()[0::2])
        loss_2 = criterion(
            (logits * scale)[1::2].view(-1),
            y.float()[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        irm_penalty = torch.sum(grad_1 * grad_2)
        return (loss_1+loss_2)/2, irm_penalty
    else:
        ele_loss = criterion((logits * scale).view(-1), y.float())
        loss= (ele_loss).mean()
        grad= autograd.grad(
            loss, [scale],
            create_graph=True)[0]
        return loss, (grad ** 2).sum()


def _irm_penalty_v1share(logits, y, criterion, model):
    # need to see the dimension of logits[::2]
    if logits.shape[0] == 0:
        return 0, 0
    if logits.shape[0] >= 2:
        loss_1 = criterion(
            (logits)[0::2].view(-1),
            y.float()[0::2])
        loss_2 = criterion(
            (logits)[1::2].view(-1),
            y.float()[1::2])
        penalty_param = list(filter(
            lambda p:id(p) in
            model.share_param_id(),
            model.parameters()))
        grad_1s = autograd.grad(loss_1,
            penalty_param, create_graph=True)
        penalty_param = list(filter(
            lambda p:id(p) in
            model.share_param_id(),
            model.parameters()))
        grad_2s = autograd.grad(loss_2,
            penalty_param, create_graph=True)
        irm_penalty = sum([
            torch.mean(grad_1 * grad_2)
            for (grad_1, grad_2)
            in list(zip(grad_1s, grad_2s))])
        return loss_1+loss_2, irm_penalty
    elif logits.shape[0] == 1:
        ele_loss = criterion((logits).view(-1), y.float())
        penalty_param = list(filter(
            lambda p:id(p) in
            model.share_param_id(),
            model.parameters()))
        grad = autograd.grad(loss, penalty_param, create_graph=True)
        irm_penalty = sum([
            0.1 * grad_1 **2
            for grad_1
            in grad])
        return loss, irm_penalty
    else:
        raise RuntimeError("Shouldn't be here")


def compute_irm_penalty_loss(yhat, y, group_idx, criterion, model, args=None):
    penalty = 0
    loss = 0
    loss_list = []
    penalty_list = []
    valid_env = 0
    for i in range(args.env_nums):
        ids = group_idx == i
        if ids.sum() > 0:
            valid_env += 1
            if args.irm_type == "irmv1":
                e_loss, e_penlaty= _irm_penalty_v1(yhat[ids], y[ids], criterion)
            if args.irm_type == "irmv1fc":
                e_loss, e_penlaty= _irm_penalty_v1share(yhat[ids], y[ids], criterion, model)
            elif args.irm_type == "rex":
                e_loss, e_penlaty=  criterion(yhat[ids].view(-1), y[ids].float()).mean(), torch.tensor(0)
            elif args.irm_type == "rvp":
                e_loss, e_penlaty=  criterion(yhat[ids].view(-1), y[ids].float()).mean(), torch.tensor(0)

            penalty += e_penlaty
            loss += e_loss
            loss_list.append(e_loss)
            penalty_list.append(e_penlaty)
    if args.irm_type == "rex":
        loss_t = torch.stack(loss_list)
        penalty = ((loss_t - loss_t.mean())** 2).mean()
    if args.irm_type == "rvp":
        loss_t = torch.stack(loss_list)
        penalty = torch.sqrt(((loss_t - loss_t.mean())** 2).mean())
    return loss, penalty

def next_batch(loader_iter, loader, dataset, batch_size=None):
    try:
        batch_data = loader_iter.__next__()
    except:
        loader_iter = iter(loader)
        batch_data = loader_iter.__next__()
    return batch_data, loader_iter

def backward_loss(loss, model, grad_clip=-1, retain_graph=False):
    loss.backward(retain_graph=retain_graph)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)


def step_rep(main_loss, optimizer, model, args):
    optimizer[0].zero_grad()
    backward_loss(main_loss, model, grad_clip=-1)
    optimizer[0].step()

def step_rep_share(main_loss, optimizer, model, args):
    if len(optimizer) >=3:
        rep_share_num = 2
    else:
        rep_share_num = 1

    [optimizer[oi].zero_grad() for oi in range(rep_share_num)]
    backward_loss(main_loss, model, grad_clip=-1)
    [optimizer[oi].step() for oi in range(rep_share_num)]

def step_share(main_loss, optimizer, model, args, retain_graph=False):
    optimizer[1].zero_grad()
    backward_loss(main_loss, model, grad_clip=-1, retain_graph=retain_graph)
    optimizer[1].step()

def step_env_share(main_loss, optimizer, model, args, retain_graph=False):
    optimizer[3].zero_grad()
    backward_loss(main_loss, model, grad_clip=-1, retain_graph=retain_graph)
    optimizer[3].step()

def step_sep(main_loss, optimizer, model, args):
    # optimizer order: rep, share, sep, rho
    optimizer[2].zero_grad()
    backward_loss(main_loss, model, grad_clip=-1)
    optimizer[2].step()

def step_env_sep(main_loss, optimizer, model, args):
    # optimizer order: rep, share, sep, rho
    optimizer[4].zero_grad()
    backward_loss(main_loss, model, grad_clip=-1)
    optimizer[4].step()

def weight_norm(model):
    weight_norm = torch.tensor(0.).cuda()
    for w in model.parameters():
        weight_norm += w.norm().pow(2)
    return weight_norm

def run_epoch_train(
    epoch, model, optimizer, loader, loader_iter, logger, args,
              is_training,scheduler=None, irm_penalty_weight=0):
    model.train()
    ees = EpochStat("Training the Epoch %s of %s" % (epoch+1), args.n_epochs)
    criterion = nn.BCEWithLogitsLoss()
    if args.irm_type in ["invrat", "irmgame"]:
        inner_loader_iter = iter(loader)
    else:
        inner_loader_iter = None
    for lr_sce in scheduler:
        lr_sce.step()

    with torch.set_grad_enabled(True):
        begin_time = time.time()
        num_steps = len(loader)
        for batch_idx in range(num_steps):
            batch, loader_iter = next_batch(loader_iter, loader, args.dataset, args.batch_size)
            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            sp = batch[3]
            if args.irm_type == "erm" or irm_penalty_weight <= 0:
                #--------other loss-------#
                model.set_sep(False)
                outputs = model(x)
                l2_loss = weight_norm(model)
                loss = criterion(outputs.view(-1), y.float())
                main_loss = loss + args.weight_decay * l2_loss
                #--------update-------#
                step_rep_share(main_loss, optimizer, model, args)
                penalty = torch.tensor(0.)
            else:
                if args.irm_type in ["rex", "irmv1",'rvp', "irmv1fc"]:
                    #--------other loss-------#
                    model.set_sep(False)
                    outputs = model(x)
                    l2_loss = weight_norm(model)
                    loss = criterion(outputs.view(-1), y.float())
                    #--------penalty-------#
                    _, penalty = compute_irm_penalty_loss(outputs, y, g, criterion, model, args)
                    main_loss = (loss + args.weight_decay* l2_loss + irm_penalty_weight * penalty) / (1 + irm_penalty_weight)
                    #--------update-------#
                    step_share(loss, optimizer, model, args, retain_graph=True)
                    step_rep(main_loss, optimizer, model, args)
                elif args.irm_type in ["invrat", "irmgame"]:
                    #----penalty loss ------#
                    model.set_sep(False)
                    xyg=(x, y, g)
                    if args.irm_type == "invrat":
                        _, penalty, inner_loader_iter = compute_irm_free_same_step_penalty_loss(y, g, model, xyg, optimizer, loader, inner_loader_iter, criterion, args)
                    elif args.irm_type == "irmgame":
                        _, penalty, inner_loader_iter = compute_irm_game_free_same_step_penalty_loss(y, g, model, xyg, optimizer, loader, inner_loader_iter, criterion, args)
                    else:
                        raise("Invalid irm_type!")
                    #-----other loss -----#
                    model.set_sep(False)
                    outputs = model(x)
                    l2_loss = weight_norm(model)
                    loss = criterion(outputs.view(-1), y.float())
                    main_loss = (loss + args.weight_decay* l2_loss + irm_penalty_weight * penalty) / (1 + irm_penalty_weight)
                    step_rep(main_loss, optimizer, model, args)
                else:
                    raise RuntimeError
                model.sep = False
            end_time = time.time()
            spend_time = end_time - begin_time
            model.set_sep(False)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            sp = batch[3]
            outputs = model(x)
            env_stats = env_stat(x, outputs, y, g, model, criterion, sp)
            del outputs
            ees.update_stats(env_stats)
            ees.update_stats(
                {"env":-1,
                 "loss": loss.item(),
                 "penalty": penalty.item(),
                 "main_loss": main_loss.item()})
        ees.echo()
        return ees.get_summary()


def run_epoch_val(epoch, model, loader, loader_iter, logger, args, val_test):
    assert val_test in ['val', 'test']
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    # ees = EpochStat("Val%s"%epoch)
    ees = EpochStat("Testing the Epoch %s of %s" % (epoch+1), args.n_epochs)
    with torch.set_grad_enabled(False):
        num_steps = len(loader)
        # print("num_steps", num_steps)
        for batch_idx in range(num_steps):
            batch, loader_iter = next_batch(loader_iter, loader, args.dataset)
            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            sp = batch[3]
            outputs = model(x)
            loss_main = criterion(outputs.view(-1), y.float())
            env_stats = env_stat(x, outputs, y, g, model, criterion, sp)
            ees.update_stats(env_stats)
        ees.echo()

def configure_optimizers(args, model):
    return model.get_optimizer_schedule(args)

def train(model, dataset,
          logger, args):

    model = model.cuda()

    optimizer, scheduler = configure_optimizers(args, model)
    best_val_acc = 0
    train_loader_iter = iter(dataset["train_loader"])
    val_loader_iter = iter(dataset["val_loader"])
    test_loader_iter = iter(dataset["test_loader"])
    for epoch in range(args.n_epochs):
        if args.irm_anneal_epochs > 0 and epoch < args.irm_anneal_epochs:
            if args.irm_anneal_type == "jump":
                irm_penalty_weight = 0
            elif args.irm_anneal_type == "linear":
                irm_penalty_weight = args.irm_penalty_weight * 1.0 * epoch / args.irm_anneal_epochs
            else:
                raise Exception
        else:
            irm_penalty_weight = args.irm_penalty_weight
        if epoch == args.irm_anneal_epochs:
            model.init_sep_by_share()
        run_epoch_train(
            epoch, model, optimizer,
            dataset['train_loader'],
            train_loader_iter,
            logger, args,
            is_training=True,
            scheduler=scheduler,
            irm_penalty_weight=irm_penalty_weight)

        run_epoch_val(
            epoch, model,
            dataset['test_loader'],
            test_loader_iter,
            logger, args,
            val_test="test")
