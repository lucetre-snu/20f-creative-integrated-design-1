import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import UNet
from dataset import *
from util import *
from loss import f1_loss, weighted_loss_and_f1_loss

import matplotlib.pyplot as plt

from torchvision import transforms, datasets


def evaluate(model, loader_val, loss_function, num_batch, current_epoch, writer_val, device):
    with torch.no_grad():
        model.eval()
        loss_arr = []
        acc_arr = []
        tp_arr = []
        tn_arr = []
        fp_arr = []
        fn_arr = []
        f1_arr = []

        for batch, data in enumerate(loader_val, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            pred = model(input)

            # 손실함수 계산하기
            loss = loss_function(pred, label)
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

            print("VALID: EPOCH %04d | BATCH %04d / %04d | LOSS %.4f | ACC %.4f | F1 SCORE %.4F" %
                (current_epoch, batch, num_batch, loss.item(), acc, f1.item()))

        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        if (current_epoch % 10) == 0:
            writer_val.add_image('label', label, current_epoch, dataformats='NHWC')
            writer_val.add_image('input', input, current_epoch, dataformats='NHWC')
            writer_val.add_image('output', pred, current_epoch, dataformats='NHWC')
        writer_val.add_scalar('loss', np.mean(loss_arr), current_epoch)
        writer_val.add_scalar('f1', np.mean(f1_arr), current_epoch)

    return np.mean(loss_arr), np.mean(acc_arr), np.mean(f1_arr), np.mean(tp_arr), np.mean(tn_arr), np.mean(fp_arr), np.mean(fn_arr)


def test_main():
    ## Parser 생성하기
    parser = argparse.ArgumentParser(description="Evaluate the UNet",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--batch_size", default=1, type=int, dest="batch_size")

    parser.add_argument("--data_dir", default="../data/final_data/", type=str, dest="data_dir")
    parser.add_argument("--ckpt_dir", default=None, type=str, dest="ckpt_dir")
    parser.add_argument("--ckpt_name", default=None, type=str, dest="ckpt_name")
    parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

    args = parser.parse_args()

    batch_size = args.batch_size
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    ckpt_name = args.ckpt_name
    result_dir = args.result_dir

    assert ckpt_dir is not None, "ckpt_dir shoud be specified"
    time_and_loss = ckpt_dir.split("/")[-1]
    if time_and_loss == "":
        time_and_loss = ckpt_dir.split("/")[-2]
    loss_name = time_and_loss.split("_")[-1]

    if ckpt_name is None:
        ckpt_lst = os.listdir(ckpt_dir)
        ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        ckpt_name = ckpt_lst[-1]

    ## 디렉토리 생성하기
    result_dir = os.path.join(result_dir, time_and_loss + "_" + ckpt_name.split(".")[0])
    if not os.path.exists(result_dir):
        os.makedirs(os.path.join(result_dir, 'png'))
        os.makedirs(os.path.join(result_dir, 'numpy'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if loss_name == "BCE":
        loss_function = nn.BCEWithLogitsLoss().to(device)
    elif loss_name == "weighted_BCE":
        loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15])).to(device)  # 237 - > 59 -> 15
    elif loss_name == "f1":
        loss_function = f1_loss
    elif loss_name =="mix":
        mix_class = weighted_loss_and_f1_loss(nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15])).to(device))
        loss_function = mix_class.loss
    else:
        assert False, loss_name + " is not supported"

    print("batch size: %d" % batch_size)
    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("ckpt_name: %s" % ckpt_name)
    print("time and loss: %s" % time_and_loss)
    print("result dir: %s" % result_dir)
    print("device: %s" % device)

    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)

    net = UNet().to(device)
    net, _, _ = load(ckpt_dir=ckpt_dir, net=net, optim=None, ckpt_name=ckpt_name)

    f = open(os.path.join(result_dir, 'log.txt'), 'w')

    with torch.no_grad():
        net.eval()
        loss_arr = []
        f1_arr = []

        for batch, data in enumerate(loader_test, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            loss = loss_function(output, label)

            loss_arr += [loss.item()]

            tp, tn, fp, fn, f1 = f1_score(output, label)
            f1_arr += [f1.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f | F1 SCORE %.4F" %
                    (batch, num_batch_test, loss.item(), f1.item()))
            f.write("TEST: BATCH %04d / %04d | LOSS %.4f | F1 SCORE %.4F\n" %
                    (batch, num_batch_test, loss.item(), f1.item()))

            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            for j in range(label.shape[0]):
                id = batch_size * (batch - 1) + j

                plt.imsave(os.path.join(result_dir, 'png', '%04d_label.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', '%04d_input.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', '%04d_output.png' % id), output[j].squeeze(), cmap='gray')

                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f | F1 SCORE %.4F" %
            (batch, num_batch_test, np.mean(loss_arr), np.mean(f1_arr)))
    f.write("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f | F1 SCORE %.4F\n" %
            (batch, num_batch_test, np.mean(loss_arr), np.mean(f1_arr)))
    f.close()

if __name__ == "__main__":
    test_main()
