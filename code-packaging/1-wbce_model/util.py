import os
import numpy as np

import torch
import torch.nn as nn

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0)


def f1_score(output, target):
    eps = torch.finfo(torch.float64).eps
    y_pred = torch.sigmoid(output)
    y_true = target
    y_pred = y_pred > 0.5

    tp = torch.sum(y_true*y_pred)
    tn = torch.sum((1-y_true)*(~y_pred))
    fp = torch.sum((1-y_true)*y_pred)
    fn = torch.sum(y_true*(~y_pred))

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
def load(ckpt_dir, net, optim, ckpt_name=None):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        print("ckpt_dir : ", ckpt_dir)
        print("ckpt_name : ", ckpt_name)
        print("failed to load model")
        return net, optim, epoch

    if ckpt_name == None:
        ckpt_lst = os.listdir(ckpt_dir)
        ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        ckpt_name = ckpt_lst[-1]

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_name))

    net.load_state_dict(dict_model['net'])
    if optim is not None:
        optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_name.split('epoch')[1].split('.pth')[0])

    return net, optim, epoch
