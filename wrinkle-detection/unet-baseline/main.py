import argparse

import os
import csv
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import UNet
from dataset import *
from util import *
from train import train
from evaluate import evaluate
from loss import f1_loss

import matplotlib.pyplot as plt

from torchvision import transforms, datasets


def main():
    ## Parser 생성하기
    parser = argparse.ArgumentParser(description="Train the UNet",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
    parser.add_argument("--batch_size", default=1, type=int, dest="batch_size")
    parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")
    parser.add_argument("--loss", default="weighted_BCE", type=str, dest="loss")

    parser.add_argument("--data_dir", default="./DATA", type=str, dest="data_dir")
    parser.add_argument("--ckpt_dir", default="./unet-baseline/checkpoint", type=str, dest="ckpt_dir")
    parser.add_argument("--log_dir", default="./unet-baseline/log", type=str, dest="log_dir")

    parser.add_argument("--mode", default="train", type=str, dest="mode")
    parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

    args = parser.parse_args()

    ## 트레이닝 파라메터 설정하기
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    loss_name = args.loss

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir

    mode = args.mode
    train_continue = args.train_continue
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    time = datetime.datetime.today()
    time = time.strftime('%m-%d_%H-%M')
    time_and_loss = time + "_" + loss_name
    ckpt_dir = os.path.join(ckpt_dir, time_and_loss)
    log_dir = os.path.join(log_dir, time_and_loss)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)
    print("loss function : %s" % loss_name)
    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("mode: %s" % mode)
    print("device: %s" % device)

    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)

    net = UNet().to(device)

    if loss_name == "BCE":
        loss_function = nn.BCEWithLogitsLoss().to(device)
    elif loss_name == "weighted_BCE":
        loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([237])).to(device)
    elif loss_name == "f1":
        loss_function = f1_loss
    else:
        assert False, loss_name + " is not supported"

    optim = torch.optim.Adam(net.parameters(), lr=lr)

    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    train_f = open(os.path.join(log_dir, 'train.tsv'), 'w', encoding='utf-8', newline='')
    train_wr = csv.writer(train_f, delimiter='\t')
    val_f = open(os.path.join(log_dir, 'val.tsv'), 'w', encoding='utf-8', newline='')
    val_wr = csv.writer(val_f, delimiter='\t')
    train_wr.writerow(["#epoch/loss/acc/f1/tp/tn/fp/fn"])
    val_wr.writerow(["#epoch/loss/acc/f1/tp/tn/fp/fn"])

    ## 네트워크 학습시키기
    st_epoch = 0

    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch, num_epoch):
        net.train()
        loss, acc, f1, tp, tn, fp, fn = train(net, loader_train, loss_function, num_batch_train, epoch, writer_train, device, optim)
        train_wr.writerow((epoch, loss, acc, f1, tp, tn, fp, fn))

        loss, acc, f1, tp, tn, fp, fn = evaluate(net, loader_val, loss_function, num_batch_val, epoch, writer_val, device)
        val_wr.writerow((epoch, loss, acc, f1, tp, tn, fp, fn))

        if (epoch % 5) == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch+1)
        print("------------------------------------------------------------")

    writer_train.close()
    writer_val.close()
    train_f.close()
    val_f.close()

if __name__ == "__main__":
    main()
