import os
import numpy as np

import torch
import torch.nn as nn

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)


def f1_score(output, target):
    eps = torch.finfo(torch.float64).eps
    y_pred = torch.sigmoid(output)
    y_true = target

    tp = torch.sum(y_true*y_pred)
    tn = torch.sum((1-y_true)*(1-y_pred))
    fp = torch.sum((1-y_true)*y_pred)
    fn = torch.sum(y_true*(1-y_pred))

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2*p*r / (p+r+eps)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return tp, tn, fp, fn, f1


## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    if optim is not None:
        optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch
