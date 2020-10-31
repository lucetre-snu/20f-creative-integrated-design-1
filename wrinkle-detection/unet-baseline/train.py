import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import UNet
from dataset import *
from util import *
from loss import *


def train(model, loader_train, loss_function, num_batch, current_epoch, writer_train, device, optim):
    model.train()
    loss_arr = []
    acc_arr = []
    tp_arr = []
    tn_arr = []
    fp_arr = []
    fn_arr = []
    f1_arr = []

    for batch, data in enumerate(loader_train, 1):
        label = data['label'].to(device)
        input = data['input'].to(device)

        pred = model(input)

        optim.zero_grad()

        loss = loss_function(pred, label)
        loss.backward()

        optim.step()

        loss_arr += [loss.item()]

        tp, tn, fp, fn, f1 = f1_score(pred, label)
        tp_arr += [tp.item()]
        tn_arr += [tn.item()]
        fp_arr += [fp.item()]
        fn_arr += [fn.item()]
        f1_arr += [f1.item()]

        label = fn_tonumpy(label)
        pred = fn_tonumpy(fn_class(pred))

        B, C, H, W = label.shape
        acc = np.sum(label == pred) / (B * C * H * W)
        acc_arr += [acc]

        print("TRAIN: EPOCH %04d | BATCH %04d / %04d | LOSS %.4f | ACC %.4f | F1 SCORE %.4F" %
            (current_epoch, batch, num_batch, loss.item(), acc, f1.item()))
        writer_train.add_scalar('loss_step', loss.item(), num_batch * current_epoch + batch)

    input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
    writer_train.add_image('label', label, current_epoch, dataformats='NHWC')
    writer_train.add_image('input', input, current_epoch, dataformats='NHWC')
    writer_train.add_image('output', pred, current_epoch, dataformats='NHWC')
    writer_train.add_scalar('loss', np.mean(loss_arr), current_epoch)

    return np.mean(loss_arr), np.mean(acc_arr), np.mean(f1_arr), np.mean(tp_arr), np.mean(tn_arr), np.mean(fp_arr), np.mean(fn_arr)


